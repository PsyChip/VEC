/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * vec-cpu - dead simple vector database (CPU mode, no GPU required)
 *
 * Usage:  vec-cpu <name> <dim> [port]
 *
 * Creates an in-memory vector store. No CUDA dependency.
 * Listens on:
 *   - TCP port (default 1920)
 *   - Windows: Named pipe \\.\pipe\vec_<name>
 *   - Linux:   Unix socket /tmp/vec_<name>.sock
 *
 * Protocol (TCP & pipe/socket):
 *   push [label] 0.1,0.2,...\n    -> slot index (text push, optional label)
 *   bpush [label]\n<dim*4 bytes>  -> slot index (binary push, optional label)
 *   pull 0.1,0.2,...\n            -> results (L2 text query)
 *   cpull 0.1,0.2,...\n           -> results (cosine text query)
 *   bpull\n<dim*4 bytes>          -> results (L2 binary query)
 *   bcpull\n<dim*4 bytes>         -> results (cosine binary query)
 *   label <index> <string>\n      -> ok (set/override label)
 *   delete <index>\n              -> ok (tombstone)
 *   undo\n                        -> ok (remove last)
 *   save\n                        -> ok (flush to disk)
 *   size\n                        -> total count
 *
 * Results format: label:distance or index:distance, comma-separated
 * Labels must not contain colons. Use URI-style paths without scheme prefix.
 * Labels saved to .meta sidecar file alongside .tensors
 *
 * File format (.tensors):
 *   [4B dim][4B count][4B deleted][1B format][count B alive mask][vector data]
 *
 * Ctrl+C saves before exit.
 *
 * Build (Windows):
 *   cl /O2 /arch:AVX2 /fp:fast /EHsc vec-cpu.cpp /Fe:vec-cpu.exe ws2_32.lib mpr.lib
 *
 * Build (Linux):
 *   g++ -O3 -march=native -ffast-math vec-cpu.cpp -o vec-cpu -lpthread
 */

/* ===================================================================== */
/*  Platform includes                                                    */
/* ===================================================================== */

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <sys/un.h>
    #include <sys/file.h>
    #include <sys/stat.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <pthread.h>
    #include <signal.h>
    #include <dirent.h>
    #include <glob.h>
    #include <fcntl.h>
    #include <errno.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

/* ===================================================================== */
/*  Constants                                                            */
/* ===================================================================== */

#define DEFAULT_PORT 1920
#define TOP_K 10
#define INITIAL_CAP 4096
#define MAX_LINE (1 << 24)
#define PIPE_BUF_SIZE (1 << 16)

#define FMT_F32 0
#define FMT_F16 1

/* ===================================================================== */
/*  CRC32                                                                */
/* ===================================================================== */

static unsigned int crc32_tab[256];
static int crc32_ready = 0;

static void crc32_init() {
    for (unsigned int i = 0; i < 256; i++) {
        unsigned int c = i;
        for (int j = 0; j < 8; j++) c = (c >> 1) ^ ((c & 1) ? 0xEDB88320 : 0);
        crc32_tab[i] = c;
    }
    crc32_ready = 1;
}

static unsigned int crc32_update(unsigned int crc, const void *buf, size_t len) {
    if (!crc32_ready) crc32_init();
    const unsigned char *p = (const unsigned char *)buf;
    crc = ~crc;
    for (size_t i = 0; i < len; i++) crc = crc32_tab[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

static const char CRC_C[] = "BCDFGHJKLMNPRSTVZ";  /* 17 */
static const char CRC_V[] = "AEIOU";              /* 5  */
#define CRC_SYL (17 * 5)  /* 85 per syllable, 85^4 = 52M unique */

static void crc32_word(unsigned int crc, char *out) {
    unsigned int v = crc;
    for (int i = 0; i < 4; i++) {
        int s = v % CRC_SYL;
        v /= CRC_SYL;
        *out++ = CRC_C[s / 5];
        *out++ = CRC_V[s % 5];
    }
    *out = '\0';
}

/* ===================================================================== */
/*  CPU distance functions                                               */
/* ===================================================================== */

static void cpu_l2_f32(const float *db, const float *query, float *dists, int n, int dim) {
    for (int i = 0; i < n; i++) {
        const float *v = db + (size_t)i * dim;
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) { float diff = v[d] - query[d]; sum += diff * diff; }
        dists[i] = sum;
    }
}

static void cpu_cos_f32(const float *db, const float *query, float *dists, int n, int dim) {
    float nq = 0;
    for (int d = 0; d < dim; d++) nq += query[d] * query[d];
    for (int i = 0; i < n; i++) {
        const float *v = db + (size_t)i * dim;
        float dot = 0, nv = 0;
        for (int d = 0; d < dim; d++) { dot += v[d]*query[d]; nv += v[d]*v[d]; }
        float denom = sqrtf(nv) * sqrtf(nq);
        dists[i] = (denom > 0) ? (1.0f - dot / denom) : 1.0f;
    }
}

/* ===================================================================== */
/*  Globals                                                              */
/* ===================================================================== */

static char g_name[256];
static char g_filepath[512];
static int g_dim = 0;
static int g_port = DEFAULT_PORT;
static int g_fmt = FMT_F32;
static int g_elem_size = 4;

static const char *fmt_name(int fmt) { return fmt == FMT_F16 ? "f16" : "f32"; }
static int g_count = 0;
static int g_capacity = 0;

static void *d_vectors = NULL;
static void *d_query = NULL;
static float *d_dists = NULL;
static float *h_dists = NULL;
static int *h_ids = NULL;
static float *h_pinned = NULL;
static int h_pinned_n = 0;

static unsigned char *g_alive = NULL;
static int g_alive_cap = 0;
static int g_deleted = 0;

/* labels: variable-length strings indexed by slot */
static char **g_labels = NULL;  /* array of pointers, NULL = no label */
static int g_labels_cap = 0;

static volatile int g_running = 1;

#ifdef _WIN32
    static HANDLE g_mutex = NULL;
#else
    static char g_sockpath[512];
    static int g_lockfd = -1;
#endif

/* ===================================================================== */
/*  Helpers                                                              */
/* ===================================================================== */

/* ===================================================================== */
/*  Memory management                                                    */
/* ===================================================================== */

static void gpu_realloc_if_needed(int required) {
    if (required <= g_capacity) return;
    int new_cap = g_capacity;
    while (new_cap < required) new_cap *= 2;

    void *d_new = malloc((size_t)new_cap * g_dim * g_elem_size);
    if (d_vectors && g_count > 0)
        memcpy(d_new, d_vectors, (size_t)g_count * g_dim * g_elem_size);
    free(d_vectors);
    d_vectors = d_new;

    free(d_dists);
    d_dists = (float *)malloc(new_cap * sizeof(float));

    free(h_dists);
    free(h_ids);
    h_dists = (float *)malloc(new_cap * sizeof(float));
    h_ids = (int *)malloc(new_cap * sizeof(int));

    unsigned char *new_alive = (unsigned char *)realloc(g_alive, new_cap);
    memset(new_alive + g_alive_cap, 1, new_cap - g_alive_cap);
    g_alive = new_alive;
    g_alive_cap = new_cap;

    char **new_labels = (char **)realloc(g_labels, new_cap * sizeof(char *));
    memset(new_labels + g_labels_cap, 0, (new_cap - g_labels_cap) * sizeof(char *));
    g_labels = new_labels;
    g_labels_cap = new_cap;

    g_capacity = new_cap;
}

static void gpu_init() {
    g_capacity = INITIAL_CAP;
    d_vectors = malloc((size_t)g_capacity * g_dim * g_elem_size);
    d_query = malloc(g_dim * g_elem_size);
    d_dists = (float *)malloc(g_capacity * sizeof(float));

    h_dists = (float *)malloc(g_capacity * sizeof(float));
    h_ids = (int *)malloc(g_capacity * sizeof(int));
    g_alive = (unsigned char *)malloc(g_capacity);
    memset(g_alive, 1, g_capacity);
    g_alive_cap = g_capacity;

    g_labels = (char **)calloc(g_capacity, sizeof(char *));
    g_labels_cap = g_capacity;

    h_pinned_n = 1024 * g_dim;
    h_pinned = (float *)malloc(h_pinned_n * sizeof(float));
}

static void gpu_ensure_pinned(int nfloats) {
    if (nfloats <= h_pinned_n) return;
    free(h_pinned);
    h_pinned_n = nfloats;
    h_pinned = (float *)malloc(h_pinned_n * sizeof(float));
}

static void gpu_shutdown() {
    free(d_vectors);
    free(d_query);
    free(d_dists);
    free(h_pinned);
    free(h_dists);
    free(h_ids);
    free(g_alive);
    if (g_labels) {
        for (int i = 0; i < g_labels_cap; i++) free(g_labels[i]);
        free(g_labels);
    }
}

/* ===================================================================== */
/*  Vector operations                                                    */
/* ===================================================================== */

static void upload_and_store(const float *h_data, void *d_dest, int nfloats) {
    memcpy(d_dest, h_data, nfloats * sizeof(float));
}

static void vec_set_label(int slot, const char *label, int len) {
    if (slot < 0 || slot >= g_labels_cap) return;
    free(g_labels[slot]);
    if (!label || len <= 0) { g_labels[slot] = NULL; return; }

    /* strip UTF-8 BOM */
    if (len >= 3 && (unsigned char)label[0] == 0xEF &&
        (unsigned char)label[1] == 0xBB && (unsigned char)label[2] == 0xBF) {
        label += 3; len -= 3;
    }

    /* trim leading/trailing whitespace */
    while (len > 0 && (*label == ' ' || *label == '\t')) { label++; len--; }
    while (len > 0 && (label[len - 1] == ' ' || label[len - 1] == '\t')) len--;
    if (len <= 0) { g_labels[slot] = NULL; return; }

    /* sanitize: escape control chars, strip colons and commas */
    char *clean = (char *)malloc(len * 2 + 1);
    int out = 0;
    int warned = 0;
    for (int i = 0; i < len; i++) {
        char c = label[i];
        if (c == '\n' || c == '\r') {
            if (c == '\r' && i + 1 < len && label[i + 1] == '\n') i++;
            clean[out++] = '\\'; clean[out++] = 'n';
        } else if (c == '\t') {
            clean[out++] = '\\'; clean[out++] = 't';
        } else if (c == ':' || c == ',') {
            if (!warned) {
                fprintf(stderr, "WARN: label for slot %d contains '%c', stripped\n", slot, c);
                warned = 1;
            }
        } else {
            clean[out++] = c;
        }
    }
    if (out > 0) {
        clean[out] = '\0';
        g_labels[slot] = clean;
    } else {
        free(clean);
        g_labels[slot] = NULL;
    }
}

static int vec_push(const float *h_vec) {
    gpu_realloc_if_needed(g_count + 1);
    int slot = g_count;
    upload_and_store(h_vec, (char *)d_vectors + (size_t)slot * g_dim * g_elem_size, g_dim);
    g_count++;
    return slot;
}

static int vec_pull(const float *h_query, int *out_ids, float *out_dists, int mode) {
    int alive = g_count - g_deleted;
    if (alive <= 0) return 0;
    int n = g_count;
    int k = (alive < TOP_K) ? alive : TOP_K;

    memcpy(d_query, h_query, g_dim * sizeof(float));

    if (mode == 1) cpu_cos_f32((const float *)d_vectors, (const float *)d_query, (float *)d_dists, n, g_dim);
    else cpu_l2_f32((const float *)d_vectors, (const float *)d_query, (float *)d_dists, n, g_dim);

    memcpy(h_dists, d_dists, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_ids[i] = i;
        if (!g_alive[i]) h_dists[i] = 3.402823e+38f;
    }

    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) {
            if (h_dists[j] < h_dists[best]) best = j;
        }
        float td = h_dists[i]; h_dists[i] = h_dists[best]; h_dists[best] = td;
        int ti = h_ids[i]; h_ids[i] = h_ids[best]; h_ids[best] = ti;
    }

    for (int i = 0; i < k; i++) {
        out_ids[i] = h_ids[i];
        out_dists[i] = h_dists[i];
    }
    return k;
}

/* ===================================================================== */
/*  Persistence                                                          */
/* ===================================================================== */

static void save_to_file() {
    if (g_count == 0) return;
    FILE *f = fopen(g_filepath, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", g_filepath); return; }

    fwrite(&g_dim, sizeof(int), 1, f);
    fwrite(&g_count, sizeof(int), 1, f);
    fwrite(&g_deleted, sizeof(int), 1, f);
    unsigned char fmt_byte = (unsigned char)g_fmt;
    fwrite(&fmt_byte, 1, 1, f);

    unsigned int crc = 0;
    if (g_count > 0) {
        fwrite(g_alive, 1, g_count, f);
        crc = crc32_update(crc, g_alive, g_count);

        size_t total_bytes = (size_t)g_count * g_dim * g_elem_size;
        fwrite(d_vectors, 1, total_bytes, f);
        crc = crc32_update(crc, d_vectors, total_bytes);
    }

    fwrite(&crc, sizeof(unsigned int), 1, f);
    fclose(f);
    char crc_name[16];
    crc32_word(crc, crc_name);
    printf("saved %d vectors to %s [%s 0x%08X]\n", g_count, g_filepath, crc_name, crc);

    /* save labels to .meta sidecar */
    int has_labels = 0;
    for (int i = 0; i < g_count; i++) { if (g_labels[i]) { has_labels = 1; break; } }
    if (has_labels) {
        char metapath[512];
        strncpy(metapath, g_filepath, sizeof(metapath) - 1);
        char *ext = strstr(metapath, ".tensors");
        if (ext) strcpy(ext, ".meta");
        else snprintf(metapath, sizeof(metapath), "%s.meta", g_name);

        FILE *mf = fopen(metapath, "wb");
        if (mf) {
            fwrite(&g_count, sizeof(int), 1, mf);
            for (int i = 0; i < g_count; i++) {
                int slen = g_labels[i] ? (int)strlen(g_labels[i]) : 0;
                fwrite(&slen, sizeof(int), 1, mf);
                if (slen > 0) fwrite(g_labels[i], 1, slen, mf);
            }
            fclose(mf);
        }
    }
}

static int peek_file_header() {
    FILE *f = fopen(g_filepath, "rb");
    if (!f) return 0;

    int file_dim;
    int dummy1, dummy2;
    unsigned char file_fmt;
    if (fread(&file_dim, sizeof(int), 1, f) != 1 ||
        fread(&dummy1, sizeof(int), 1, f) != 1 ||
        fread(&dummy2, sizeof(int), 1, f) != 1 ||
        fread(&file_fmt, 1, 1, f) != 1) {
        fclose(f);
        return 0;
    }
    fclose(f);

    if (g_dim > 0 && file_dim != g_dim)
        fprintf(stderr, "WARN: filename suggests dim=%d but header has dim=%d, using header\n", g_dim, file_dim);
    if (file_fmt != g_fmt)
        fprintf(stderr, "WARN: filename suggests %s but header has %s, using header\n", fmt_name(g_fmt), fmt_name(file_fmt));

    g_dim = file_dim;
    g_fmt = file_fmt;
    g_elem_size = (g_fmt == FMT_F16) ? 2 : 4;
    return 1;
}

static unsigned int g_loaded_crc = 0;
static unsigned int g_computed_crc = 0;
static int g_crc_ok = 0;

static int load_from_file() {
    FILE *f = fopen(g_filepath, "rb");
    if (!f) return 0;

    int file_dim, file_count, file_deleted;
    unsigned char file_fmt;
    if (fread(&file_dim, sizeof(int), 1, f) != 1 ||
        fread(&file_count, sizeof(int), 1, f) != 1 ||
        fread(&file_deleted, sizeof(int), 1, f) != 1 ||
        fread(&file_fmt, 1, 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "WARN: corrupt %s, starting fresh\n", g_filepath);
        return 0;
    }

    unsigned int crc = 0;
    if (file_count > 0) {
        gpu_realloc_if_needed(file_count);
        size_t mask_rd = fread(g_alive, 1, file_count, f);
        if ((int)mask_rd != file_count) {
            fprintf(stderr, "WARN: alive mask truncated\n");
            file_count = (int)mask_rd;
        }
        crc = crc32_update(crc, g_alive, mask_rd);

        size_t total_bytes = (size_t)file_count * g_dim * g_elem_size;
        void *h_buf = malloc(total_bytes);
        size_t rd = fread(h_buf, 1, total_bytes, f);
        if (rd != total_bytes) {
            fprintf(stderr, "WARN: data truncated\n");
            file_count = (int)(rd / (g_dim * g_elem_size));
        }
        crc = crc32_update(crc, h_buf, rd);
        memcpy(d_vectors, h_buf, (size_t)file_count * g_dim * g_elem_size);
        free(h_buf);
        g_count = file_count;
        g_deleted = file_deleted;
    }

    unsigned int stored_crc = 0;
    if (fread(&stored_crc, sizeof(unsigned int), 1, f) == 1) {
        g_loaded_crc = stored_crc;
        g_computed_crc = crc;
        g_crc_ok = (stored_crc == crc);
    } else {
        g_loaded_crc = 0;
        g_computed_crc = crc;
        g_crc_ok = -1; /* no trailer - old format file */
    }

    fclose(f);

    /* load labels from .meta sidecar */
    char metapath[512];
    strncpy(metapath, g_filepath, sizeof(metapath) - 1);
    char *mext = strstr(metapath, ".tensors");
    if (mext) strcpy(mext, ".meta");
    else snprintf(metapath, sizeof(metapath), "%s.meta", g_name);

    FILE *mf = fopen(metapath, "rb");
    if (mf) {
        int meta_count = 0;
        if (fread(&meta_count, sizeof(int), 1, mf) == 1) {
            int lim = (meta_count < g_count) ? meta_count : g_count;
            for (int i = 0; i < lim; i++) {
                int slen = 0;
                if (fread(&slen, sizeof(int), 1, mf) != 1) break;
                if (slen > 0 && slen < 65536) {
                    g_labels[i] = (char *)malloc(slen + 1);
                    if (fread(g_labels[i], 1, slen, mf) != (size_t)slen) { free(g_labels[i]); g_labels[i] = NULL; break; }
                    g_labels[i][slen] = '\0';
                } else if (slen == 0) {
                    /* skip - no label */
                } else {
                    break; /* corrupt */
                }
            }
        }
        fclose(mf);
    }

    return 1;
}

/* ===================================================================== */
/*  Protocol parser                                                      */
/* ===================================================================== */

static int parse_floats(const char *s, float *out, int max_n) {
    int n = 0;
    const char *p = s;
    while (*p && n < max_n) {
        char *end;
        float v = strtof(p, &end);
        if (end == p) break;
        out[n++] = v;
        p = end;
        if (*p == ',') p++;
    }
    return n;
}

typedef int (*write_fn)(void *ctx, const char *buf, int len);

/* format results: label:dist or index:dist per result, comma-separated */
static void format_results(int *ids, float *dists, int k, write_fn writer, void *wctx) {
    char resp[65536];
    char *p = resp;
    int rem = sizeof(resp) - 2;
    for (int i = 0; i < k; i++) {
        int w;
        const char *lbl = (ids[i] < g_labels_cap) ? g_labels[ids[i]] : NULL;
        if (lbl)
            w = snprintf(p, rem, "%s%s:%.6f", i > 0 ? "," : "", lbl, dists[i]);
        else
            w = snprintf(p, rem, "%s%d:%.6f", i > 0 ? "," : "", ids[i], dists[i]);
        p += w; rem -= w;
    }
    *p++ = '\n';
    writer(wctx, resp, (int)(p - resp));
}

static int process_command(const char *line, int line_len, write_fn writer, void *wctx, const char *bin_payload, int bin_payload_len) {
    char resp[4096];
    int rlen;

    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) line_len--;
    if (line_len == 0) return 0;

    /* push [label] floats  OR  push floats */
    /* push ["quoted label"|label] floats */
    if (line_len > 5 && strncmp(line, "push ", 5) == 0) {
        const char *data = line + 5;
        int data_len = line_len - 5;
        const char *label = NULL;
        int label_len = 0;
        const char *floats_start = data;

        if (*data == '"') {
            /* quoted label: push "some text here" 0.12,0.23,... */
            const char *close = (const char *)memchr(data + 1, '"', data_len - 1);
            if (close) {
                label = data + 1;
                label_len = (int)(close - data - 1);
                floats_start = close + 1;
                while (*floats_start == ' ') floats_start++;
            }
        } else {
            /* unquoted: scan for first float-like token with a comma */
            for (const char *s = data; s < data + data_len; s++) {
                if ((*s >= '0' && *s <= '9') || *s == '-' || *s == '.') {
                    const char *t = s;
                    while (t < data + data_len && *t != ' ' && *t != '\n') t++;
                    for (const char *c = s; c < t; c++) {
                        if (*c == ',') { floats_start = s; goto found_push; }
                    }
                    if (t >= data + data_len) { floats_start = s; goto found_push; }
                }
            }
        found_push:
            if (floats_start > data) {
                label = data;
                label_len = (int)(floats_start - data);
                while (label_len > 0 && label[label_len - 1] == ' ') label_len--;
            }
        }

        float *vals = (float *)malloc(g_dim * sizeof(float));
        int n = parse_floats(floats_start, vals, g_dim);
        if (n != g_dim) {
            rlen = snprintf(resp, sizeof(resp), "err dim mismatch: got %d, expected %d\n", n, g_dim);
            writer(wctx, resp, rlen);
            free(vals);
            return 0;
        }
        int slot = vec_push(vals);
        free(vals);
        if (label_len > 0) vec_set_label(slot, label, label_len);
        rlen = snprintf(resp, sizeof(resp), "%d\n", slot);
        writer(wctx, resp, rlen);
        return 0;
    }

    /* bpush [label]\n<dim*4 bytes> */
    if (line_len >= 5 && strncmp(line, "bpush", 5) == 0) {
        int expected_bytes = g_dim * (int)sizeof(float);
        if (bin_payload_len != expected_bytes) {
            rlen = snprintf(resp, sizeof(resp), "err need %d bytes, got %d\n", expected_bytes, bin_payload_len);
            writer(wctx, resp, rlen);
            return 0;
        }
        /* label is anything after "bpush " - quoted or unquoted */
        const char *label = NULL;
        int label_len = 0;
        if (line_len > 6) {
            const char *lstart = line + 6;
            int llen = line_len - 6;
            while (llen > 0 && (lstart[llen - 1] == ' ' || lstart[llen - 1] == '\r')) llen--;
            if (llen > 0 && lstart[0] == '"' && lstart[llen - 1] == '"') {
                label = lstart + 1;
                label_len = llen - 2;
            } else {
                label = lstart;
                label_len = llen;
            }
        }
        float *vals = (float *)malloc(expected_bytes);
        memcpy(vals, bin_payload, expected_bytes);
        int slot = vec_push(vals);
        free(vals);
        if (label_len > 0) vec_set_label(slot, label, label_len);
        rlen = snprintf(resp, sizeof(resp), "%d\n", slot);
        writer(wctx, resp, rlen);
        return 0;
    }

    /* pull (L2 text), cpull (cosine text) */
    int pull_mode = -1;
    int pull_offset = 0;
    if (line_len > 5 && strncmp(line, "pull ", 5) == 0) { pull_mode = 0; pull_offset = 5; }
    else if (line_len > 6 && strncmp(line, "cpull ", 6) == 0) { pull_mode = 1; pull_offset = 6; }
    if (pull_mode >= 0) {
        float *vals = (float *)malloc(g_dim * sizeof(float));
        int n = parse_floats(line + pull_offset, vals, g_dim);
        if (n != g_dim) {
            rlen = snprintf(resp, sizeof(resp), "err dim mismatch: got %d, expected %d\n", n, g_dim);
            writer(wctx, resp, rlen);
            free(vals);
            return 0;
        }
        int ids[TOP_K];
        float dists[TOP_K];
        int k = vec_pull(vals, ids, dists, pull_mode);
        free(vals);
        format_results(ids, dists, k, writer, wctx);
        return 0;
    }

    /* bpull (L2 binary), bcpull (cosine binary) */
    int bpull_mode = -1;
    if (line_len >= 5 && strncmp(line, "bpull", 5) == 0 && (line_len == 5 || line[5] == '\r')) bpull_mode = 0;
    else if (line_len >= 6 && strncmp(line, "bcpull", 6) == 0 && (line_len == 6 || line[6] == '\r')) bpull_mode = 1;
    if (bpull_mode >= 0) {
        int expected_bytes = g_dim * (int)sizeof(float);
        if (bin_payload_len != expected_bytes) {
            rlen = snprintf(resp, sizeof(resp), "err need %d bytes, got %d\n", expected_bytes, bin_payload_len);
            writer(wctx, resp, rlen);
            return 0;
        }
        float *vals = (float *)malloc(expected_bytes);
        memcpy(vals, bin_payload, expected_bytes);
        int ids[TOP_K];
        float dists[TOP_K];
        int k = vec_pull(vals, ids, dists, bpull_mode);
        free(vals);
        format_results(ids, dists, k, writer, wctx);
        return 0;
    }

    /* label <index> <string> */
    /* label <index> ["quoted"|unquoted] */
    if (line_len > 6 && strncmp(line, "label ", 6) == 0) {
        const char *p = line + 6;
        int idx = atoi(p);
        while (*p && *p != ' ') p++;
        if (*p == ' ') p++;
        int lbl_len = (int)(line + line_len - p);
        if (idx < 0 || idx >= g_count) {
            rlen = snprintf(resp, sizeof(resp), "err index out of range\n");
            writer(wctx, resp, rlen);
            return 0;
        }
        if (lbl_len >= 2 && p[0] == '"' && p[lbl_len - 1] == '"') { p++; lbl_len -= 2; }
        vec_set_label(idx, lbl_len > 0 ? p : NULL, lbl_len);
        rlen = snprintf(resp, sizeof(resp), "ok\n");
        writer(wctx, resp, rlen);
        return 0;
    }

    if (line_len > 7 && strncmp(line, "delete ", 7) == 0) {
        int idx = atoi(line + 7);
        if (idx < 0 || idx >= g_count) {
            rlen = snprintf(resp, sizeof(resp), "err index out of range\n");
            writer(wctx, resp, rlen);
            return 0;
        }
        if (!g_alive[idx]) {
            rlen = snprintf(resp, sizeof(resp), "err already deleted\n");
            writer(wctx, resp, rlen);
            return 0;
        }
        g_alive[idx] = 0;
        g_deleted++;
        rlen = snprintf(resp, sizeof(resp), "ok\n");
        writer(wctx, resp, rlen);
        return 0;
    }

    if (line_len >= 4 && strncmp(line, "undo", 4) == 0) {
        if (g_count == 0) {
            rlen = snprintf(resp, sizeof(resp), "err empty\n");
            writer(wctx, resp, rlen);
            return 0;
        }
        g_count--;
        if (!g_alive[g_count]) g_deleted--;
        g_alive[g_count] = 1;
        vec_set_label(g_count, NULL, 0);
        rlen = snprintf(resp, sizeof(resp), "ok\n");
        writer(wctx, resp, rlen);
        return 0;
    }

    if (line_len >= 4 && strncmp(line, "save", 4) == 0) {
        save_to_file();
        rlen = snprintf(resp, sizeof(resp), "ok\n");
        writer(wctx, resp, rlen);
        return 0;
    }

    if (line_len >= 4 && strncmp(line, "size", 4) == 0) {
        rlen = snprintf(resp, sizeof(resp), "%d\n", g_count);
        writer(wctx, resp, rlen);
        return 0;
    }

    rlen = snprintf(resp, sizeof(resp), "err unknown command\n");
    writer(wctx, resp, rlen);
    return 0;
}

/* ===================================================================== */
/*  Argument parsing (shared)                                            */
/* ===================================================================== */

static void set_format(const char *s) {
    if (strcmp(s, "f16") == 0 || strcmp(s, "fp16") == 0 || strcmp(s, "16") == 0) { g_fmt = FMT_F16; g_elem_size = 2; }
    else if (strcmp(s, "f32") == 0 || strcmp(s, "fp32") == 0 || strcmp(s, "32") == 0) { g_fmt = FMT_F32; g_elem_size = 4; }
    else { fprintf(stderr, "ERROR: unknown format '%s' (use f16 or f32)\n", s); exit(1); }
}

static void parse_dim_format(const char *arg) {
    char buf[256];
    strncpy(buf, arg, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    char *colon = strchr(buf, ':');
    if (colon) {
        *colon = '\0';
        set_format(colon + 1);
    }
    g_dim = atoi(buf);
}

static void build_filepath() {
    snprintf(g_filepath, sizeof(g_filepath), "%s_%d_%s.tensors", g_name, g_dim, fmt_name(g_fmt));
}

static int parse_tensors_filename(const char *filename) {
    const char *base = filename;
    const char *p = filename;
    while (*p) {
        if (*p == '\\' || *p == '/') base = p + 1;
        p++;
    }
    char buf[256];
    strncpy(buf, base, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *ext = strstr(buf, ".tensors");
    if (!ext) return 0;
    *ext = '\0';

    char *last_sep = strrchr(buf, '_');
    if (!last_sep) return 0;
    *last_sep = '\0';
    const char *fmt_str = last_sep + 1;

    char *dim_sep = strrchr(buf, '_');
    if (!dim_sep) return 0;
    *dim_sep = '\0';
    const char *dim_str = dim_sep + 1;

    strncpy(g_name, buf, sizeof(g_name) - 1);
    g_dim = atoi(dim_str);
    set_format(fmt_str);

    return (g_dim > 0);
}

/* ===================================================================== */
/*  Startup / capacity display (shared)                                  */
/* ===================================================================== */

static void fmt_count(char *buf, size_t sz, int count) {
    if (count >= 1000000) snprintf(buf, sz, "%.1fm", count / 1000000.0);
    else if (count >= 1000) snprintf(buf, sz, "%.1fk", count / 1000.0);
    else snprintf(buf, sz, "%d", count);
}

static void fmt_bytes(char *buf, size_t sz, double bytes) {
    if (bytes >= 1073741824.0) snprintf(buf, sz, "%.1f GB", bytes / 1073741824.0);
    else if (bytes >= 1048576.0) snprintf(buf, sz, "%.1f MB", bytes / 1048576.0);
    else if (bytes >= 1024.0) snprintf(buf, sz, "%.1f KB", bytes / 1024.0);
    else snprintf(buf, sz, "%.0f B", bytes);
}

static void fmt_modified(char *buf, size_t sz, time_t mtime) {
    static const char *months[] = {"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};
    time_t now = time(NULL);
    long diff = (long)(now - mtime);
    struct tm *ft = localtime(&mtime);
    int fh = ft->tm_hour, fm = ft->tm_min;
    int fday = ft->tm_mday, fmon = ft->tm_mon, fyear = 1900 + ft->tm_year;

    if (diff < 60) {
        snprintf(buf, sz, "just now (%02d:%02d)", fh, fm);
    } else if (diff < 3600) {
        snprintf(buf, sz, "%ld minutes ago (%02d:%02d)", diff / 60, fh, fm);
    } else if (diff < 86400) {
        snprintf(buf, sz, "%ld hours ago (%02d:%02d)", diff / 3600, fh, fm);
    } else if (diff < 172800) {
        snprintf(buf, sz, "yesterday %02d:%02d", fh, fm);
    } else {
        snprintf(buf, sz, "%d %s %d (%ld days ago)", fday, months[fmon], fyear, diff / 86400);
    }
}

static void print_startup_info(int file_exists, int loaded, double max_records) {
    printf("===================================================================\n");

    if (file_exists && loaded > 0) {
        char cnt[32], active[32], del[32], cap[32], rem[32], fsz[32], ago[64];
        int alive = g_count - g_deleted;
        double remaining = max_records - g_count;
        double avail = (remaining / max_records) * 100.0;
        if (avail < 0) { avail = 0; remaining = 0; }

        fmt_count(cnt, sizeof(cnt), g_count);
        fmt_count(active, sizeof(active), alive);
        fmt_count(del, sizeof(del), g_deleted);
        fmt_count(cap, sizeof(cap), (int)max_records);
        fmt_count(rem, sizeof(rem), (int)remaining);

        double file_bytes = (double)g_count * g_dim * g_elem_size + (double)g_count + 13 + 4;
        fmt_bytes(fsz, sizeof(fsz), file_bytes);

        /* get file modification time */
#ifdef _WIN32
        WIN32_FILE_ATTRIBUTE_DATA fattr;
        time_t mtime = 0;
        if (GetFileAttributesExA(g_filepath, GetFileExInfoStandard, &fattr)) {
            ULARGE_INTEGER ull;
            ull.LowPart = fattr.ftLastWriteTime.dwLowDateTime;
            ull.HighPart = fattr.ftLastWriteTime.dwHighDateTime;
            mtime = (time_t)((ull.QuadPart - 116444736000000000ULL) / 10000000ULL);
        }
#else
        struct stat st;
        time_t mtime = 0;
        if (stat(g_filepath, &st) == 0) mtime = st.st_mtime;
#endif

        char date_str[128] = "unknown";
        if (mtime > 0) fmt_modified(date_str, sizeof(date_str), mtime);

        printf("  database      %s\n", g_name);
        printf("  format        %s, %d dim\n", fmt_name(g_fmt), g_dim);
        printf("  records       %s total, %s active, %s deleted\n", cnt, active, del);
        printf("  file size     %s\n", fsz);
        printf("  modified      %s\n", date_str);
        printf("  capacity      %s max, %s remaining (%.1f%%)\n", cap, rem, avail);

        char crc_name[16];
        if (g_crc_ok == 1) {
            crc32_word(g_loaded_crc, crc_name);
            printf("  checksum      %s (0x%08X) ok\n", crc_name, g_loaded_crc);
        } else if (g_crc_ok == 0)
            printf("  checksum      MISMATCH - expected 0x%08X, got 0x%08X\n", g_loaded_crc, g_computed_crc);
        else
            printf("  checksum      none (old format)\n");
    } else {
        char cap[32];
        fmt_count(cap, sizeof(cap), (int)max_records);
        printf("  database      %s (new)\n", g_name);
        printf("  format        %s, %d dim\n", fmt_name(g_fmt), g_dim);
        printf("  capacity      %s records max\n", cap);
    }

    printf("===================================================================\n");
}

/* ===================================================================== */
/*  Help text                                                            */
/* ===================================================================== */

static void print_help() {
    printf("vec - dead simple GPU-resident vector database\n\n");
    printf("usage:\n");
    printf("  vec                              auto-detect or create new db\n");
    printf("  vec <name> [dim[:fmt]] [port]    create/open database\n");
    printf("  vec <file.tensors> [port]        load from file\n");
    printf("  vec --notcp <name> [dim] [port]  run without TCP (pipe/socket only)\n");
    printf("  vec --route <port>               router mode (forward TCP to pipes)\n");
    printf("  vec --help                       this message\n\n");
    printf("protocol:\n");
    printf("  push [label] <floats>            store vector with optional label\n");
    printf("  bpush [label]\\n<bytes>           binary push (raw fp32)\n");
    printf("  pull <floats>                    query L2 distance\n");
    printf("  cpull <floats>                   query cosine distance\n");
    printf("  bpull\\n<bytes>                   binary L2 query\n");
    printf("  bcpull\\n<bytes>                  binary cosine query\n");
    printf("  label <idx> <string>             set/override label\n");
    printf("  delete <idx>                     tombstone vector\n");
    printf("  undo                             remove last push\n");
    printf("  save                             flush to disk\n");
    printf("  size                             total record count\n\n");
    printf("router mode:\n");
    printf("  start instances with --notcp, then route them through one port:\n");
    printf("  vec --notcp tools 1024\n");
    printf("  vec --notcp conversations 1024\n");
    printf("  vec --route 1920\n");
    printf("  client sends: push tools 0.12,...\\n\n");
}

/* ===================================================================== */
/*  Router mode                                                          */
/* ===================================================================== */

#define MAX_ROUTES 64

struct route_entry {
    char name[64];
#ifdef _WIN32
    HANDLE pipe;
#else
    int fd;
#endif
    int connected;
};

static route_entry g_routes[MAX_ROUTES];
static int g_route_count = 0;

#ifdef _WIN32

static void router_discover_pipes() {
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA("\\\\.\\pipe\\*", &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (strncmp(fd.cFileName, "vec_", 4) != 0) continue;
        if (g_route_count >= MAX_ROUTES) break;
        route_entry *r = &g_routes[g_route_count];
        memset(r, 0, sizeof(*r));
        strncpy(r->name, fd.cFileName + 4, sizeof(r->name) - 1);
        r->pipe = INVALID_HANDLE_VALUE;
        r->connected = 0;
        g_route_count++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static int router_connect(route_entry *r) {
    if (r->connected) return 1;
    char pipename[512];
    snprintf(pipename, sizeof(pipename), "\\\\.\\pipe\\vec_%s", r->name);
    r->pipe = CreateFileA(pipename, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (r->pipe == INVALID_HANDLE_VALUE) return 0;
    DWORD mode = PIPE_READMODE_BYTE;
    SetNamedPipeHandleState(r->pipe, &mode, NULL, NULL);
    r->connected = 1;
    return 1;
}

static int router_send_recv(route_entry *r, const char *data, int data_len, char *resp, int resp_max) {
    if (!router_connect(r)) return -1;
    DWORD written;
    if (!WriteFile(r->pipe, data, data_len, &written, NULL)) {
        CloseHandle(r->pipe); r->pipe = INVALID_HANDLE_VALUE; r->connected = 0; return -1;
    }
    FlushFileBuffers(r->pipe);
    int total = 0;
    while (total < resp_max - 1) {
        DWORD bytesRead;
        if (!ReadFile(r->pipe, resp + total, 1, &bytesRead, NULL) || bytesRead == 0) break;
        total++;
        if (resp[total - 1] == '\n') break;
    }
    resp[total] = '\0';
    return total;
}

static DWORD WINAPI router_client_thread(LPVOID param) {
    SOCKET client = *(SOCKET *)param;
    free(param);
    char buf[65536];
    int buf_used = 0;
    while (1) {
        int r = recv(client, buf + buf_used, sizeof(buf) - buf_used - 1, 0);
        if (r <= 0) break;
        buf_used += r;
        buf[buf_used] = '\0';
        while (1) {
            char *nl = (char *)memchr(buf, '\n', buf_used);
            if (!nl) break;
            int line_len = (int)(nl - buf);
            /* parse: command namespace [args...] */
            int cmd_len = 0;
            while (cmd_len < line_len && buf[cmd_len] != ' ') cmd_len++;
            const char *ns_start = buf + cmd_len + 1;
            int ns_len = 0;
            while (cmd_len + 1 + ns_len < line_len && ns_start[ns_len] != ' ') ns_len++;
            route_entry *route = NULL;
            if (ns_len > 0) {
                for (int i = 0; i < g_route_count; i++) {
                    if ((int)strlen(g_routes[i].name) == ns_len && strncmp(g_routes[i].name, ns_start, ns_len) == 0) { route = &g_routes[i]; break; }
                }
                if (!route) {
                    int old_count = g_route_count;
                    router_discover_pipes();
                    for (int i = old_count; i < g_route_count; i++) {
                        if ((int)strlen(g_routes[i].name) == ns_len && strncmp(g_routes[i].name, ns_start, ns_len) == 0) { route = &g_routes[i]; break; }
                    }
                }
            }
            char resp[65536];
            if (!route) {
                int rlen = snprintf(resp, sizeof(resp), "err unknown namespace '%.*s'\n", ns_len, ns_start);
                send(client, resp, rlen, 0);
            } else {
                /* rebuild as: command [args...]\n (strip namespace) */
                char fwd[65536];
                int fwd_len = 0;
                memcpy(fwd, buf, cmd_len);
                fwd_len = cmd_len;
                int rest_off = cmd_len + 1 + ns_len;
                if (rest_off < line_len) {
                    memcpy(fwd + fwd_len, buf + rest_off, line_len - rest_off);
                    fwd_len += line_len - rest_off;
                }
                fwd[fwd_len] = '\n';
                int rlen = router_send_recv(route, fwd, fwd_len + 1, resp, sizeof(resp));
                if (rlen > 0) send(client, resp, rlen, 0);
                else { int elen = snprintf(resp, sizeof(resp), "err pipe disconnected '%s'\n", route->name); send(client, resp, elen, 0); }
            }
            int consumed = line_len + 1;
            buf_used -= consumed;
            if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
        }
    }
    closesocket(client);
    return 0;
}

static int run_router(int port) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    router_discover_pipes();
    printf("vec router on port %d\n", port);
    printf("discovered %d namespace%s:\n", g_route_count, g_route_count == 1 ? "" : "s");
    for (int i = 0; i < g_route_count; i++) printf("  %s -> \\\\.\\pipe\\vec_%s\n", g_routes[i].name, g_routes[i].name);
    if (g_route_count == 0) printf("  (none found, will discover on first request)\n");
    printf("\n");
    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)port);
    if (bind(listen_sock, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "ERROR: cannot bind port %d\n", port); closesocket(listen_sock); WSACleanup(); return 1;
    }
    listen(listen_sock, SOMAXCONN);
    printf("ready for connections\n");
    while (1) {
        SOCKET client = accept(listen_sock, NULL, NULL);
        if (client == INVALID_SOCKET) continue;
        SOCKET *ps = (SOCKET *)malloc(sizeof(SOCKET));
        *ps = client;
        CreateThread(NULL, 0, router_client_thread, ps, 0, NULL);
    }
    closesocket(listen_sock); WSACleanup(); return 0;
}

#else /* Linux router */

static void router_discover_sockets() {
    glob_t gl;
    if (glob("/tmp/vec_*.sock", 0, NULL, &gl) != 0) return;
    for (size_t i = 0; i < gl.gl_pathc && g_route_count < MAX_ROUTES; i++) {
        const char *path = gl.gl_pathv[i];
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (strncmp(base, "vec_", 4) != 0) continue;
        const char *name_start = base + 4;
        const char *dot = strstr(name_start, ".sock");
        if (!dot) continue;
        int name_len = (int)(dot - name_start);
        route_entry *r = &g_routes[g_route_count];
        memset(r, 0, sizeof(*r));
        memcpy(r->name, name_start, name_len < 63 ? name_len : 63);
        r->fd = -1;
        r->connected = 0;
        g_route_count++;
    }
    globfree(&gl);
}

static int router_connect(route_entry *r) {
    if (r->connected) return 1;
    r->fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (r->fd < 0) return 0;
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "/tmp/vec_%s.sock", r->name);
    if (connect(r->fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) { close(r->fd); r->fd = -1; return 0; }
    r->connected = 1;
    return 1;
}

static int router_send_recv(route_entry *r, const char *data, int data_len, char *resp, int resp_max) {
    if (!router_connect(r)) return -1;
    if (send(r->fd, data, data_len, 0) <= 0) { close(r->fd); r->fd = -1; r->connected = 0; return -1; }
    int total = 0;
    while (total < resp_max - 1) {
        int rd = recv(r->fd, resp + total, 1, 0);
        if (rd <= 0) break;
        total++;
        if (resp[total - 1] == '\n') break;
    }
    resp[total] = '\0';
    return total;
}

static void *router_client_thread(void *param) {
    int client = *(int *)param;
    free(param);
    char buf[65536];
    int buf_used = 0;
    while (1) {
        int r = recv(client, buf + buf_used, sizeof(buf) - buf_used - 1, 0);
        if (r <= 0) break;
        buf_used += r;
        buf[buf_used] = '\0';
        while (1) {
            char *nl = (char *)memchr(buf, '\n', buf_used);
            if (!nl) break;
            int line_len = (int)(nl - buf);
            /* parse: command namespace [args...] */
            int cmd_len = 0;
            while (cmd_len < line_len && buf[cmd_len] != ' ') cmd_len++;
            const char *ns_start = buf + cmd_len + 1;
            int ns_len = 0;
            while (cmd_len + 1 + ns_len < line_len && ns_start[ns_len] != ' ') ns_len++;
            route_entry *route = NULL;
            if (ns_len > 0) {
                for (int i = 0; i < g_route_count; i++) {
                    if ((int)strlen(g_routes[i].name) == ns_len && strncmp(g_routes[i].name, ns_start, ns_len) == 0) { route = &g_routes[i]; break; }
                }
                if (!route) {
                    int old_count = g_route_count;
                    router_discover_sockets();
                    for (int i = old_count; i < g_route_count; i++) {
                        if ((int)strlen(g_routes[i].name) == ns_len && strncmp(g_routes[i].name, ns_start, ns_len) == 0) { route = &g_routes[i]; break; }
                    }
                }
            }
            char resp[65536];
            if (!route) {
                int rlen = snprintf(resp, sizeof(resp), "err unknown namespace '%.*s'\n", ns_len, ns_start);
                send(client, resp, rlen, 0);
            } else {
                /* rebuild as: command [args...]\n (strip namespace) */
                char fwd[65536];
                int fwd_len = 0;
                memcpy(fwd, buf, cmd_len);
                fwd_len = cmd_len;
                int rest_off = cmd_len + 1 + ns_len;
                if (rest_off < line_len) {
                    memcpy(fwd + fwd_len, buf + rest_off, line_len - rest_off);
                    fwd_len += line_len - rest_off;
                }
                fwd[fwd_len] = '\n';
                int rlen = router_send_recv(route, fwd, fwd_len + 1, resp, sizeof(resp));
                if (rlen > 0) send(client, resp, rlen, 0);
                else { int elen = snprintf(resp, sizeof(resp), "err socket disconnected '%s'\n", route->name); send(client, resp, elen, 0); }
            }
            int consumed = line_len + 1;
            buf_used -= consumed;
            if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
        }
    }
    close(client);
    return NULL;
}

static int run_router(int port) {
    router_discover_sockets();
    printf("vec router on port %d\n", port);
    printf("discovered %d namespace%s:\n", g_route_count, g_route_count == 1 ? "" : "s");
    for (int i = 0; i < g_route_count; i++) printf("  %s -> /tmp/vec_%s.sock\n", g_routes[i].name, g_routes[i].name);
    if (g_route_count == 0) printf("  (none found, will discover on first request)\n");
    printf("\n");
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)port);
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "ERROR: cannot bind port %d\n", port); close(listen_fd); return 1;
    }
    listen(listen_fd, SOMAXCONN);
    printf("ready for connections\n");
    while (1) {
        int client = accept(listen_fd, NULL, NULL);
        if (client < 0) continue;
        int *pc = (int *)malloc(sizeof(int));
        *pc = client;
        pthread_t t;
        pthread_create(&t, NULL, router_client_thread, pc);
        pthread_detach(t);
    }
    close(listen_fd); return 0;
}

#endif

/* ######################################################################
 * ##                                                                  ##
 * ##  WINDOWS PLATFORM CODE                                           ##
 * ##                                                                  ##
 * ###################################################################### */

#ifdef _WIN32

static int acquire_instance_lock() {
    char mutex_name[512];
    snprintf(mutex_name, sizeof(mutex_name), "Global\\vec_%s", g_name);
    g_mutex = CreateMutexA(NULL, TRUE, mutex_name);
    if (g_mutex == NULL) {
        fprintf(stderr, "ERROR: failed to create mutex: %lu\n", GetLastError());
        return 0;
    }
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        fprintf(stderr, "ERROR: database '%s' is already running\n", g_name);
        CloseHandle(g_mutex);
        g_mutex = NULL;
        return 0;
    }
    return 1;
}

static void release_instance_lock() {
    if (g_mutex) {
        ReleaseMutex(g_mutex);
        CloseHandle(g_mutex);
        g_mutex = NULL;
    }
}

static void generate_random_name(char *buf, int len) {
    const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    srand((unsigned)GetTickCount());
    for (int i = 0; i < len; i++) buf[i] = chars[rand() % (sizeof(chars) - 1)];
    buf[len] = '\0';
}

static int find_any_db() {
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA("*.tensors", &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;

    int best_score = -1;

    do {
        char buf[256];
        strncpy(buf, fd.cFileName, sizeof(buf) - 1);
        char *ext = strstr(buf, ".tensors");
        if (!ext) continue;
        *ext = '\0';

        char *ls = strrchr(buf, '_');
        if (!ls) continue;
        *ls = '\0';
        const char *fs = ls + 1;

        char *ds = strrchr(buf, '_');
        if (!ds) continue;
        *ds = '\0';

        int d = atoi(ds + 1);
        if (d <= 0) continue;

        int fv = (strcmp(fs, "f16") == 0) ? FMT_F16 : FMT_F32;

        int score = 0;
        if (fv == FMT_F32) score += 100;
        if (d == 1024) score += 50;

        if (score > best_score) {
            best_score = score;
            strncpy(g_name, buf, sizeof(g_name) - 1);
            g_dim = d;
            if (fv == FMT_F16) { g_fmt = FMT_F16; g_elem_size = 2; }
            else { g_fmt = FMT_F32; g_elem_size = 4; }
        }
    } while (FindNextFileA(h, &fd));

    FindClose(h);
    if (best_score >= 0) {
        build_filepath();
        return 1;
    }
    return 0;
}

static int find_existing_db(const char *name) {
    WIN32_FIND_DATAA fd;
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s_*.tensors", name);

    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;

    int best_score = -1;

    do {
        char buf[256];
        strncpy(buf, fd.cFileName, sizeof(buf) - 1);
        char *ext = strstr(buf, ".tensors");
        if (!ext) continue;
        *ext = '\0';

        char *ls = strrchr(buf, '_');
        if (!ls) continue;
        *ls = '\0';
        const char *fs = ls + 1;

        char *ds = strrchr(buf, '_');
        if (!ds) continue;
        *ds = '\0';

        int d = atoi(ds + 1);
        if (d <= 0) continue;

        int fv = (strcmp(fs, "f16") == 0) ? FMT_F16 : FMT_F32;

        int score = 0;
        if (fv == FMT_F32) score += 100;
        if (d == 1024) score += 50;

        if (score > best_score) {
            best_score = score;
            strncpy(g_name, buf, sizeof(g_name) - 1);
            g_dim = d;
            if (fv == FMT_F16) { g_fmt = FMT_F16; g_elem_size = 2; }
            else { g_fmt = FMT_F32; g_elem_size = 4; }
        }
    } while (FindNextFileA(h, &fd));

    FindClose(h);
    if (best_score >= 0) {
        build_filepath();
        return 1;
    }
    return 0;
}

/* ----- TCP (Windows) ------------------------------------------------- */

static int tcp_writer(void *ctx, const char *buf, int len) {
    SOCKET s = *(SOCKET *)ctx;
    return send(s, buf, len, 0);
}

static DWORD WINAPI tcp_client_thread(LPVOID param) {
    SOCKET client = *(SOCKET *)param;
    free(param);
    char *buf = (char *)malloc(MAX_LINE);
    int buf_used = 0;

    while (g_running) {
        int r = recv(client, buf + buf_used, MAX_LINE - buf_used - 1, 0);
        if (r <= 0) break;
        buf_used += r;
        buf[buf_used] = '\0';

        while (1) {
            char *nl = (char *)memchr(buf, '\n', buf_used);
            if (!nl) break;
            int line_len = (int)(nl - buf);

            int is_binary = (line_len >= 5 && strncmp(buf, "bpush", 5) == 0) ||
                            (line_len >= 5 && strncmp(buf, "bpull", 5) == 0) ||
                            (line_len >= 6 && strncmp(buf, "bcpull", 6) == 0);
            if (is_binary) {
                int payload_bytes = g_dim * (int)sizeof(float);
                int header_bytes = line_len + 1;
                int total_needed = header_bytes + payload_bytes;
                while (buf_used < total_needed && g_running) {
                    r = recv(client, buf + buf_used, MAX_LINE - buf_used, 0);
                    if (r <= 0) goto done;
                    buf_used += r;
                }
                int rc = process_command(buf, line_len, tcp_writer, &client, buf + header_bytes, payload_bytes);
                int consumed = total_needed;
                buf_used -= consumed;
                if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                if (rc == 1) goto done;
            } else {
                int rc = process_command(buf, line_len, tcp_writer, &client, NULL, 0);
                int consumed = line_len + 1;
                buf_used -= consumed;
                if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                if (rc == 1) goto done;
            }
        }
    }

done:
    closesocket(client);
    free(buf);
    return 0;
}

static DWORD WINAPI tcp_listener_thread(LPVOID param) {
    (void)param;
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_sock == INVALID_SOCKET) {
        fprintf(stderr, "ERROR: socket() failed: %d\n", WSAGetLastError());
        g_running = 0;
        return 1;
    }

    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)g_port);

    if (bind(listen_sock, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        int err = WSAGetLastError();
        if (err == WSAEADDRINUSE) fprintf(stderr, "ERROR: port %d is already in use\n", g_port);
        else fprintf(stderr, "ERROR: bind() failed: %d\n", err);
        closesocket(listen_sock);
        g_running = 0;
        return 1;
    }

    listen(listen_sock, SOMAXCONN);
    printf("TCP listening on 0.0.0.0:%d\n", g_port);

    while (g_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_sock, &fds);
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        int sel = select(0, &fds, NULL, NULL, &tv);
        if (sel <= 0) continue;
        SOCKET client = accept(listen_sock, NULL, NULL);
        if (client == INVALID_SOCKET) continue;
        SOCKET *ps = (SOCKET *)malloc(sizeof(SOCKET));
        *ps = client;
        CreateThread(NULL, 0, tcp_client_thread, ps, 0, NULL);
    }

    closesocket(listen_sock);
    WSACleanup();
    return 0;
}

/* ----- Named Pipe (Windows) ------------------------------------------ */

static int pipe_writer(void *ctx, const char *buf, int len) {
    HANDLE pipe = *(HANDLE *)ctx;
    DWORD written;
    WriteFile(pipe, buf, len, &written, NULL);
    FlushFileBuffers(pipe);
    return (int)written;
}

static DWORD WINAPI pipe_listener_thread(LPVOID param) {
    (void)param;
    char pipe_name[512];
    snprintf(pipe_name, sizeof(pipe_name), "\\\\.\\pipe\\vec_%s", g_name);
    printf("Pipe listening on %s\n", pipe_name);

    while (g_running) {
        HANDLE pipe = CreateNamedPipeA(pipe_name, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, PIPE_UNLIMITED_INSTANCES,
            PIPE_BUF_SIZE, PIPE_BUF_SIZE, 1000, NULL);

        if (pipe == INVALID_HANDLE_VALUE) {
            fprintf(stderr, "ERROR: CreateNamedPipe failed: %lu\n", GetLastError());
            Sleep(1000);
            continue;
        }

        OVERLAPPED ov;
        memset(&ov, 0, sizeof(ov));
        ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        ConnectNamedPipe(pipe, &ov);

        BOOL connected = FALSE;
        while (g_running) {
            DWORD wait = WaitForSingleObject(ov.hEvent, 1000);
            if (wait == WAIT_OBJECT_0) { connected = TRUE; break; }
        }
        CloseHandle(ov.hEvent);

        if (!connected || !g_running) {
            DisconnectNamedPipe(pipe);
            CloseHandle(pipe);
            if (!g_running) break;
            continue;
        }

        char *buf = (char *)malloc(MAX_LINE);
        int buf_used = 0;

        while (g_running) {
            DWORD bytesRead;
            BOOL ok = ReadFile(pipe, buf + buf_used, MAX_LINE - buf_used - 1, &bytesRead, NULL);
            if (!ok || bytesRead == 0) break;
            buf_used += (int)bytesRead;
            buf[buf_used] = '\0';

            while (1) {
                char *nl = (char *)memchr(buf, '\n', buf_used);
                if (!nl) break;
                int line_len = (int)(nl - buf);

                int is_binary = (line_len >= 5 && strncmp(buf, "bpush", 5) == 0) ||
                                (line_len >= 5 && strncmp(buf, "bpull", 5) == 0) ||
                                (line_len >= 6 && strncmp(buf, "bcpull", 6) == 0);
                if (is_binary) {
                    int payload_bytes = g_dim * (int)sizeof(float);
                    int header_bytes = line_len + 1;
                    int total_needed = header_bytes + payload_bytes;
                    while (buf_used < total_needed && g_running) {
                        ok = ReadFile(pipe, buf + buf_used, MAX_LINE - buf_used, &bytesRead, NULL);
                        if (!ok || bytesRead == 0) goto pipe_done;
                        buf_used += (int)bytesRead;
                    }
                    int rc = process_command(buf, line_len, pipe_writer, &pipe, buf + header_bytes, payload_bytes);
                    int consumed = total_needed;
                    buf_used -= consumed;
                    if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                    if (rc == 1) goto pipe_done;
                } else {
                    int rc = process_command(buf, line_len, pipe_writer, &pipe, NULL, 0);
                    int consumed = line_len + 1;
                    buf_used -= consumed;
                    if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                    if (rc == 1) goto pipe_done;
                }
            }
        }

    pipe_done:
        free(buf);
        FlushFileBuffers(pipe);
        DisconnectNamedPipe(pipe);
        CloseHandle(pipe);
    }
    return 0;
}

/* ----- Signal + Main (Windows) --------------------------------------- */

static BOOL WINAPI ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        if (!g_running) return TRUE;
        g_running = 0;
        printf("\nshutting down...\n");
        save_to_file();
        return TRUE;
    }
    return FALSE;
}

int main(int argc, char **argv) {
    /* chdir to exe directory so file discovery works from shortcuts */
    {
        char exepath[512];
        GetModuleFileNameA(NULL, exepath, sizeof(exepath));
        char *last = strrchr(exepath, '\\');
        if (last) { *last = '\0'; SetCurrentDirectoryA(exepath); }
    }

    /* flags */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
        if (strcmp(argv[i], "--route") == 0) { return run_router((i + 1 < argc) ? atoi(argv[i + 1]) : 1920); }
    }

    int no_tcp = 0;
    if (argc > 1 && strcmp(argv[1], "--notcp") == 0) { no_tcp = 1; argc--; argv++; }

    int port_arg_idx = -1;

    if (argc < 2) {
        if (!find_any_db()) {
            generate_random_name(g_name, 6);
            g_dim = 1024;
            build_filepath();
        }
    } else {
        const char *arg1 = argv[1];

        if (strstr(arg1, ".tensors")) {
            strncpy(g_filepath, arg1, sizeof(g_filepath) - 1);
            const char *base = arg1;
            for (const char *p = arg1; *p; p++)
                if (*p == '\\' || *p == '/') base = p + 1;
            char tmp[256];
            strncpy(tmp, base, sizeof(tmp) - 1);
            char *dot = strstr(tmp, ".tensors");
            if (dot) *dot = '\0';
            strncpy(g_name, tmp, sizeof(g_name) - 1);
            FILE *check = fopen(g_filepath, "rb");
            if (!check) {
                fprintf(stderr, "ERROR: file not found: %s\n", g_filepath);
                return 1;
            }
            fclose(check);
            if (!peek_file_header()) {
                fprintf(stderr, "ERROR: cannot read header from %s\n", g_filepath);
                return 1;
            }
            port_arg_idx = 2;
        } else if (argc >= 3) {
            strncpy(g_name, arg1, sizeof(g_name) - 1);
            parse_dim_format(argv[2]);
            build_filepath();
            port_arg_idx = 3;
        } else {
            strncpy(g_name, arg1, sizeof(g_name) - 1);
            if (!find_existing_db(arg1)) {
                g_dim = 1024;
                build_filepath();
            }
            port_arg_idx = 2;
        }
    }

    if (port_arg_idx > 0 && port_arg_idx < argc) {
        g_port = atoi(argv[port_arg_idx]);
        if (g_port <= 0 || g_port > 65535) {
            fprintf(stderr, "ERROR: port must be between 1 and 65535\n");
            return 1;
        }
    }

    if (g_dim <= 0 || g_dim > 65536) {
        fprintf(stderr, "ERROR: dimension must be between 1 and 65536\n");
        return 1;
    }

    if (!acquire_instance_lock()) return 1;

    int file_exists = peek_file_header();

    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    double total_ram_gb = ms.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
    double max_records = (ms.ullTotalPhys * 0.8) / ((double)g_dim * g_elem_size);

    printf("vec-cpu (%.1f GB RAM)\n", total_ram_gb);
    if (total_ram_gb < 8.0)
        fprintf(stderr, "WARN: only %.1f GB RAM available\n", total_ram_gb);

    if (g_fmt == FMT_F16) {
        fprintf(stderr, "ERROR: fp16 requires GPU. Use fp32 with vec-cpu.\n");
        release_instance_lock();
        return 1;
    }

    gpu_init();

    int lr = load_from_file();
    if (lr < 0) {
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }

    print_startup_info(file_exists, lr, max_records);

    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    HANDLE h_tcp = NULL;
    if (!no_tcp) h_tcp = CreateThread(NULL, 0, tcp_listener_thread, NULL, 0, NULL);
    HANDLE h_pipe = CreateThread(NULL, 0, pipe_listener_thread, NULL, 0, NULL);

    Sleep(200);
    if (!g_running) {
        if (h_tcp) WaitForSingleObject(h_tcp, 2000);
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }
    printf("ready for connections, ctrl+c to exit\n");

    while (g_running) { Sleep(500); }

    if (h_tcp) WaitForSingleObject(h_tcp, 3000);
    WaitForSingleObject(h_pipe, 3000);

    gpu_shutdown();
    release_instance_lock();
    return 0;
}

/* ######################################################################
 * ##                                                                  ##
 * ##  LINUX PLATFORM CODE                                             ##
 * ##                                                                  ##
 * ###################################################################### */

#else /* !_WIN32 */

static int acquire_instance_lock() {
    char lockpath[512];
    snprintf(lockpath, sizeof(lockpath), "/tmp/vec_%s.lock", g_name);
    g_lockfd = open(lockpath, O_CREAT | O_RDWR, 0644);
    if (g_lockfd < 0) {
        fprintf(stderr, "ERROR: cannot create lock file: %s\n", strerror(errno));
        return 0;
    }
    if (flock(g_lockfd, LOCK_EX | LOCK_NB) != 0) {
        fprintf(stderr, "ERROR: database '%s' is already running\n", g_name);
        close(g_lockfd);
        g_lockfd = -1;
        return 0;
    }
    return 1;
}

static void release_instance_lock() {
    if (g_lockfd >= 0) {
        flock(g_lockfd, LOCK_UN);
        close(g_lockfd);
        char lockpath[512];
        snprintf(lockpath, sizeof(lockpath), "/tmp/vec_%s.lock", g_name);
        unlink(lockpath);
        g_lockfd = -1;
    }
}

static void generate_random_name(char *buf, int len) {
    srand((unsigned)time(NULL));
    const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    for (int i = 0; i < len; i++) buf[i] = chars[rand() % (sizeof(chars) - 1)];
    buf[len] = '\0';
}

static int find_any_db() {
    glob_t gl;
    if (glob("*.tensors", 0, NULL, &gl) != 0) return 0;

    int best_score = -1;

    for (size_t i = 0; i < gl.gl_pathc; i++) {
        const char *path = gl.gl_pathv[i];
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;

        char buf[256];
        strncpy(buf, base, sizeof(buf) - 1);
        char *ext = strstr(buf, ".tensors");
        if (!ext) continue;
        *ext = '\0';

        char *ls = strrchr(buf, '_');
        if (!ls) continue;
        *ls = '\0';
        const char *fs = ls + 1;

        char *ds = strrchr(buf, '_');
        if (!ds) continue;
        *ds = '\0';

        int d = atoi(ds + 1);
        if (d <= 0) continue;

        int fv = (strcmp(fs, "f16") == 0) ? FMT_F16 : FMT_F32;

        int score = 0;
        if (fv == FMT_F32) score += 100;
        if (d == 1024) score += 50;

        if (score > best_score) {
            best_score = score;
            strncpy(g_name, buf, sizeof(g_name) - 1);
            g_dim = d;
            if (fv == FMT_F16) { g_fmt = FMT_F16; g_elem_size = 2; }
            else { g_fmt = FMT_F32; g_elem_size = 4; }
        }
    }

    globfree(&gl);
    if (best_score >= 0) {
        build_filepath();
        return 1;
    }
    return 0;
}

static int find_existing_db(const char *name) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s_*.tensors", name);

    glob_t gl;
    if (glob(pattern, 0, NULL, &gl) != 0) return 0;

    int best_score = -1;

    for (size_t i = 0; i < gl.gl_pathc; i++) {
        const char *path = gl.gl_pathv[i];
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;

        char buf[256];
        strncpy(buf, base, sizeof(buf) - 1);
        char *ext = strstr(buf, ".tensors");
        if (!ext) continue;
        *ext = '\0';

        char *ls = strrchr(buf, '_');
        if (!ls) continue;
        *ls = '\0';
        const char *fs = ls + 1;

        char *ds = strrchr(buf, '_');
        if (!ds) continue;
        *ds = '\0';

        int d = atoi(ds + 1);
        if (d <= 0) continue;

        int fv = (strcmp(fs, "f16") == 0) ? FMT_F16 : FMT_F32;

        int score = 0;
        if (fv == FMT_F32) score += 100;
        if (d == 1024) score += 50;

        if (score > best_score) {
            best_score = score;
            strncpy(g_name, buf, sizeof(g_name) - 1);
            g_dim = d;
            if (fv == FMT_F16) { g_fmt = FMT_F16; g_elem_size = 2; }
            else { g_fmt = FMT_F32; g_elem_size = 4; }
        }
    }

    globfree(&gl);
    if (best_score >= 0) {
        build_filepath();
        return 1;
    }
    return 0;
}

/* ----- TCP (Linux) --------------------------------------------------- */

static int sock_writer(void *ctx, const char *buf, int len) {
    int fd = *(int *)ctx;
    return send(fd, buf, len, 0);
}

static void *client_thread(void *param) {
    int client = *(int *)param;
    free(param);
    char *buf = (char *)malloc(MAX_LINE);
    int buf_used = 0;

    while (g_running) {
        int r = recv(client, buf + buf_used, MAX_LINE - buf_used - 1, 0);
        if (r <= 0) break;
        buf_used += r;
        buf[buf_used] = '\0';

        while (1) {
            char *nl = (char *)memchr(buf, '\n', buf_used);
            if (!nl) break;
            int line_len = (int)(nl - buf);

            int is_binary = (line_len >= 5 && strncmp(buf, "bpush", 5) == 0) ||
                            (line_len >= 5 && strncmp(buf, "bpull", 5) == 0) ||
                            (line_len >= 6 && strncmp(buf, "bcpull", 6) == 0);
            if (is_binary) {
                int payload_bytes = g_dim * (int)sizeof(float);
                int header_bytes = line_len + 1;
                int total_needed = header_bytes + payload_bytes;
                while (buf_used < total_needed && g_running) {
                    r = recv(client, buf + buf_used, MAX_LINE - buf_used, 0);
                    if (r <= 0) goto done;
                    buf_used += r;
                }
                int rc = process_command(buf, line_len, sock_writer, &client, buf + header_bytes, payload_bytes);
                int consumed = total_needed;
                buf_used -= consumed;
                if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                if (rc == 1) goto done;
            } else {
                int rc = process_command(buf, line_len, sock_writer, &client, NULL, 0);
                int consumed = line_len + 1;
                buf_used -= consumed;
                if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                if (rc == 1) goto done;
            }
        }
    }

done:
    close(client);
    free(buf);
    return NULL;
}

static void *tcp_listener_thread(void *param) {
    (void)param;
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "ERROR: socket() failed: %s\n", strerror(errno));
        g_running = 0;
        return NULL;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)g_port);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        if (errno == EADDRINUSE) fprintf(stderr, "ERROR: port %d is already in use\n", g_port);
        else fprintf(stderr, "ERROR: bind() failed: %s\n", strerror(errno));
        close(listen_fd);
        g_running = 0;
        return NULL;
    }

    listen(listen_fd, SOMAXCONN);
    printf("TCP listening on 0.0.0.0:%d\n", g_port);

    while (g_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_fd, &fds);
        struct timeval tv = {1, 0};
        if (select(listen_fd + 1, &fds, NULL, NULL, &tv) <= 0) continue;
        int client = accept(listen_fd, NULL, NULL);
        if (client < 0) continue;
        int *pc = (int *)malloc(sizeof(int));
        *pc = client;
        pthread_t t;
        pthread_create(&t, NULL, client_thread, pc);
        pthread_detach(t);
    }

    close(listen_fd);
    return NULL;
}

/* ----- Unix Domain Socket (Linux) ------------------------------------ */

static void *unix_listener_thread(void *param) {
    (void)param;
    unlink(g_sockpath);

    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "ERROR: unix socket failed: %s\n", strerror(errno));
        return NULL;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_sockpath, sizeof(addr.sun_path) - 1);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "ERROR: unix bind failed: %s\n", strerror(errno));
        close(listen_fd);
        return NULL;
    }

    listen(listen_fd, SOMAXCONN);
    printf("Socket listening on %s\n", g_sockpath);

    while (g_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_fd, &fds);
        struct timeval tv = {1, 0};
        if (select(listen_fd + 1, &fds, NULL, NULL, &tv) <= 0) continue;
        int client = accept(listen_fd, NULL, NULL);
        if (client < 0) continue;
        int *pc = (int *)malloc(sizeof(int));
        *pc = client;
        pthread_t t;
        pthread_create(&t, NULL, client_thread, pc);
        pthread_detach(t);
    }

    close(listen_fd);
    unlink(g_sockpath);
    return NULL;
}

/* ----- Signal + Main (Linux) ----------------------------------------- */

static void sig_handler(int sig) {
    (void)sig;
    if (!g_running) return;
    g_running = 0;
    printf("\nshutting down...\n");
    save_to_file();
}

int main(int argc, char **argv) {
    /* chdir to exe directory so file discovery works from symlinks */
    {
        char exepath[512] = {0};
        ssize_t len = readlink("/proc/self/exe", exepath, sizeof(exepath) - 1);
        if (len <= 0) {
            char *resolved = realpath(argv[0], NULL);
            if (resolved) { strncpy(exepath, resolved, sizeof(exepath) - 1); free(resolved); }
        } else {
            exepath[len] = '\0';
        }
        char *last = strrchr(exepath, '/');
        if (last) { *last = '\0'; chdir(exepath); }
    }

    /* flags */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
        if (strcmp(argv[i], "--route") == 0) { return run_router((i + 1 < argc) ? atoi(argv[i + 1]) : 1920); }
    }

    int no_tcp = 0;
    if (argc > 1 && strcmp(argv[1], "--notcp") == 0) { no_tcp = 1; argc--; argv++; }

    int port_arg_idx = -1;

    if (argc < 2) {
        if (!find_any_db()) {
            generate_random_name(g_name, 6);
            g_dim = 1024;
            build_filepath();
        }
    } else {
        const char *arg1 = argv[1];

        if (strstr(arg1, ".tensors")) {
            strncpy(g_filepath, arg1, sizeof(g_filepath) - 1);
            const char *base = arg1;
            for (const char *p = arg1; *p; p++)
                if (*p == '/') base = p + 1;
            char tmp[256];
            strncpy(tmp, base, sizeof(tmp) - 1);
            char *dot = strstr(tmp, ".tensors");
            if (dot) *dot = '\0';
            strncpy(g_name, tmp, sizeof(g_name) - 1);
            FILE *check = fopen(g_filepath, "rb");
            if (!check) {
                fprintf(stderr, "ERROR: file not found: %s\n", g_filepath);
                return 1;
            }
            fclose(check);
            if (!peek_file_header()) {
                fprintf(stderr, "ERROR: cannot read header from %s\n", g_filepath);
                return 1;
            }
            port_arg_idx = 2;
        } else if (argc >= 3) {
            strncpy(g_name, arg1, sizeof(g_name) - 1);
            parse_dim_format(argv[2]);
            build_filepath();
            port_arg_idx = 3;
        } else {
            strncpy(g_name, arg1, sizeof(g_name) - 1);
            if (!find_existing_db(arg1)) {
                g_dim = 1024;
                build_filepath();
            }
            port_arg_idx = 2;
        }
    }

    if (port_arg_idx > 0 && port_arg_idx < argc) {
        g_port = atoi(argv[port_arg_idx]);
        if (g_port <= 0 || g_port > 65535) {
            fprintf(stderr, "ERROR: port must be between 1 and 65535\n");
            return 1;
        }
    }

    if (g_dim <= 0 || g_dim > 65536) {
        fprintf(stderr, "ERROR: dimension must be between 1 and 65536\n");
        return 1;
    }

    if (!acquire_instance_lock()) return 1;

    snprintf(g_sockpath, sizeof(g_sockpath), "/tmp/vec_%s.sock", g_name);

    int file_exists = peek_file_header();

    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    double total_ram_gb = (double)pages * page_size / (1024.0 * 1024.0 * 1024.0);
    double max_records = (total_ram_gb * 1024.0 * 1024.0 * 1024.0 * 0.8) / ((double)g_dim * g_elem_size);

    printf("vec-cpu (%.1f GB RAM)\n", total_ram_gb);
    if (total_ram_gb < 8.0)
        fprintf(stderr, "WARN: only %.1f GB RAM available\n", total_ram_gb);

    if (g_fmt == FMT_F16) {
        fprintf(stderr, "ERROR: fp16 requires GPU. Use fp32 with vec-cpu.\n");
        release_instance_lock();
        return 1;
    }

    gpu_init();

    int lr = load_from_file();
    if (lr < 0) {
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }

    print_startup_info(file_exists, lr, max_records);

    struct sigaction sa;
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    pthread_t t_tcp, t_unix;
    int has_tcp = 0;
    if (!no_tcp) { pthread_create(&t_tcp, NULL, tcp_listener_thread, NULL); has_tcp = 1; }
    pthread_create(&t_unix, NULL, unix_listener_thread, NULL);

    usleep(200000);
    if (!g_running) {
        if (has_tcp) pthread_join(t_tcp, NULL);
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }
    printf("ready for connections, ctrl+c to exit\n");

    while (g_running) { usleep(500000); }

    if (has_tcp) pthread_join(t_tcp, NULL);
    pthread_join(t_unix, NULL);

    gpu_shutdown();
    release_instance_lock();
    return 0;
}

#endif /* _WIN32 */
