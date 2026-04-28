/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * vec - dead simple GPU-resident vector database
 *
 * Usage:  vec <name> <dim[:format]> [port]
 *         vec deploy [port]
 *         vec --deploy=name:dim[:fmt][,...] [port]
 *         format: f32 (default), f16
 *
 * Creates an in-memory CUDA vector store.
 * Listens on:
 *   - TCP port (default 1920)
 *   - Windows: Named pipe \\.\pipe\vec_<name>
 *   - Linux:   Unix socket /tmp/vec_<name>.sock
 *
 * VEC 2.0 binary frame protocol — see PROTOCOL-2.0.md for the byte-exact spec.
 *   request:  F0 <2B ns_len> [ns] <CMD> <2B label_len> [label] <4B body_len> [body]
 *   response: <1B status> <4B body_len> [body]   ; status 0=OK, 1=ERR
 *   CMD: 01=push 02=query 04=get 06=update 07=delete 08=label
 *        09=undo 0A=save 0D=cluster 0E=distinct 0F=represent
 *        10=info 11=qid 13=set_data 14=get_data
 *
 * GPU top-K kernel for datasets above 100K entries.
 * Labels: filename-scheme, ≤2048 bytes, validated on write, lenient on load.
 * Data:   opaque blobs ≤100KB, requires a label.
 *
 * File format:
 *   .tensors  [4B dim][4B count][4B deleted][1B fmt][count B alive][vectors][4B CRC32]
 *   .meta     [4B count][per slot: 4B len + label bytes]
 *   .data     [4B count][count B alive mask][per present slot: 4B len + bytes][4B CRC32]
 *
 * Features:
 *   - Read-only mode if file is not writable
 *   - Disk space check before save
 *   - File integrity check/repair (--check, --repair)
 *
 * Ctrl+C saves before exit.
 *
 * Build (Windows):
 *   nvcc -O2 -c vec_kernel.cu -o vec_kernel.obj <gencode flags>
 *   nvcc -O2 vec_kernel.obj vec.cpp -o vec.exe -lws2_32 <gencode flags>
 *
 * Build (Linux):
 *   nvcc -O2 -c vec_kernel.cu -o vec_kernel.o <gencode flags>
 *   nvcc -O2 vec_kernel.o vec.cpp -o vec -lpthread <gencode flags>
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
    #include <sys/statvfs.h>
    #include <sys/wait.h>
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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

/* ===================================================================== */
/*  Constants                                                            */
/* ===================================================================== */

#define DEFAULT_PORT 1920
#define DEFAULT_TOP_K 10
#define GPU_TOP_K 10
#define GPU_SORT_THRESHOLD 100000
#define INITIAL_CAP 4096
#define MAX_LINE (1 << 24)
#define PIPE_BUF_SIZE (1 << 20)

#define FMT_F32 0
#define FMT_F16 1

/* VEC 2.0 binary frame protocol — see PROTOCOL-2.0.md */
#define BIN_MAGIC        0xF0
#define PROTOCOL_VERSION 0x02

#define CMD_PUSH      0x01
#define CMD_QUERY     0x02   /* unified PULL+CPULL */
/*      0x03            removed — was CPULL */
#define CMD_GET       0x04   /* unified GET+MGET */
/*      0x05            removed — was MGET */
#define CMD_UPDATE    0x06
#define CMD_DELETE    0x07
#define CMD_LABEL     0x08
#define CMD_UNDO      0x09
#define CMD_SAVE      0x0A
/*      0x0B, 0x0C      reserved (do not reuse) */
#define CMD_CLUSTER   0x0D
#define CMD_DISTINCT  0x0E
#define CMD_REPRESENT 0x0F
#define CMD_INFO      0x10
#define CMD_QID       0x11   /* unified PID+CPID */
/*      0x12            removed — was CPID */
#define CMD_SET_DATA  0x13
#define CMD_GET_DATA  0x14

/* response envelope */
#define RESP_OK  0x00
#define RESP_ERR 0x01

/* shape mask bits */
#define SHAPE_VECTOR 0x01
#define SHAPE_LABEL  0x02
#define SHAPE_DATA   0x04

/* GET mode */
#define GET_MODE_SINGLE 0x00
#define GET_MODE_BATCH  0x01

/* metric */
#define METRIC_L2     0x00
#define METRIC_COSINE 0x01

#ifdef _WIN32
#define strtok_r strtok_s
#endif

/* ===================================================================== */
/*  Elapsed-time helper                                                  */
/* ===================================================================== */

/* monotonic timestamp in ms since some arbitrary epoch */
static long long now_ms() {
#ifdef _WIN32
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (long long)((c.QuadPart * 1000LL) / f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
#endif
}

/* format elapsed ms into a compact human string: "23ms", "4.1s", "52m", "2h13m" */
static void format_elapsed(long long ms, char *out, int outsz) {
    if (ms < 1000) {
        snprintf(out, outsz, "%lldms", ms);
    } else if (ms < 60000) {
        snprintf(out, outsz, "%.1fs", ms / 1000.0);
    } else if (ms < 3600000) {
        long long m = ms / 60000;
        long long s = (ms % 60000) / 1000;
        snprintf(out, outsz, "%lldm%llds", m, s);
    } else {
        long long h = ms / 3600000;
        long long m = (ms % 3600000) / 60000;
        snprintf(out, outsz, "%lldh%lldm", h, m);
    }
}

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
/*  External kernels (vec_kernel.cu)                                     */
/* ===================================================================== */

extern "C" {
    void launch_l2_f32(const float *db, const float *query, float *dists, int n, int dim);
    void launch_cos_f32(const float *db, const float *query, float *dists, int n, int dim);
    void launch_l2_f16(const void *db, const void *query, float *dists, int n, int dim);
    void launch_cos_f16(const void *db, const void *query, float *dists, int n, int dim);
    void launch_f32_to_f16(const float *src, void *dst, int count);
    void launch_topk(const float *d_dists, const unsigned char *d_alive, int n,
                      float *out_dists, int *out_ids);
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
static float *d_topk_dists = NULL;
static int *d_topk_ids = NULL;
static unsigned char *d_alive = NULL;
static float *d_staging = NULL;
static int d_staging_n = 0;

static float *h_dists = NULL;
static int *h_ids = NULL;

static unsigned char *g_alive = NULL;
static int g_alive_cap = 0;
static int g_deleted = 0;

/* labels: variable-length strings indexed by slot */
static char **g_labels = NULL;  /* array of pointers, NULL = no label */
static int g_labels_cap = 0;

/* data blobs: variable-length opaque payloads indexed by slot, ≤ MAX_DATA_BYTES each */
#define MAX_DATA_BYTES 102400
#define MAX_LABEL_BYTES 2048
static unsigned char **g_blobs = NULL;  /* array of pointers, NULL = no blob */
static unsigned int  *g_blob_lens = NULL; /* 0 if g_blobs[i] == NULL */
static int g_blobs_cap = 0;

static int g_dirty = 0;
static int g_readonly = 0;
static volatile int g_running = 1;
static volatile time_t g_last_write = 0;
#define AUTOSAVE_IDLE_SECS 60

#ifdef _WIN32
    static HANDLE g_mutex = NULL;
    static CRITICAL_SECTION g_req_mutex;
    #define REQ_LOCK()   EnterCriticalSection(&g_req_mutex)
    #define REQ_UNLOCK() LeaveCriticalSection(&g_req_mutex)
#else
    static char g_sockpath[512];
    static int g_lockfd = -1;
    static pthread_mutex_t g_req_mutex = PTHREAD_MUTEX_INITIALIZER;
    #define REQ_LOCK()   pthread_mutex_lock(&g_req_mutex)
    #define REQ_UNLOCK() pthread_mutex_unlock(&g_req_mutex)
#endif

/* ===================================================================== */
/*  CUDA helpers                                                         */
/* ===================================================================== */

static void save_to_file(int already_locked); /* forward declaration for CUDA_CHECK */

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "FATAL: CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        save_to_file(0); \
        exit(1); \
    } \
} while(0)

/* ===================================================================== */
/*  GPU memory management                                                */
/* ===================================================================== */

static void gpu_ensure_staging(int nfloats) {
    if (nfloats <= d_staging_n) return;
    if (d_staging) CUDA_CHECK(cudaFree(d_staging));
    d_staging_n = nfloats;
    CUDA_CHECK(cudaMalloc(&d_staging, d_staging_n * sizeof(float)));
}

static void gpu_realloc_if_needed(int required) {
    if (required <= g_capacity) return;
    int new_cap = g_capacity;
    while (new_cap < required) new_cap *= 2;

    void *d_new;
    CUDA_CHECK(cudaMalloc(&d_new, (size_t)new_cap * g_dim * g_elem_size));
    if (d_vectors && g_count > 0) {
        CUDA_CHECK(cudaMemcpy(d_new, d_vectors, (size_t)g_count * g_dim * g_elem_size, cudaMemcpyDeviceToDevice));
    }
    if (d_vectors) CUDA_CHECK(cudaFree(d_vectors));
    d_vectors = d_new;

    if (d_dists) CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaMalloc(&d_dists, new_cap * sizeof(float)));

    free(h_dists);
    free(h_ids);
    h_dists = (float *)malloc(new_cap * sizeof(float));
    h_ids   = (int *)  malloc(new_cap * sizeof(int));
    if (!h_dists || !h_ids) {
        fprintf(stderr, "FATAL: out of CPU memory allocating distance buffers (%d slots)\n", new_cap);
        save_to_file(1);
        exit(1);
    }

    unsigned char *new_alive = (unsigned char *)realloc(g_alive, new_cap);
    if (!new_alive) {
        fprintf(stderr, "FATAL: out of CPU memory reallocating alive mask (%d bytes)\n", new_cap);
        save_to_file(1);
        exit(1);
    }
    memset(new_alive + g_alive_cap, 1, new_cap - g_alive_cap);
    g_alive = new_alive;
    g_alive_cap = new_cap;

    if (d_alive) CUDA_CHECK(cudaFree(d_alive));
    CUDA_CHECK(cudaMalloc(&d_alive, new_cap));
    CUDA_CHECK(cudaMemcpy(d_alive, g_alive, new_cap, cudaMemcpyHostToDevice));

    char **new_labels = (char **)realloc(g_labels, new_cap * sizeof(char *));
    if (!new_labels) {
        fprintf(stderr, "FATAL: out of CPU memory reallocating label table (%d slots)\n", new_cap);
        save_to_file(1);
        exit(1);
    }
    memset(new_labels + g_labels_cap, 0, (new_cap - g_labels_cap) * sizeof(char *));
    g_labels = new_labels;
    g_labels_cap = new_cap;

    unsigned char **new_blobs = (unsigned char **)realloc(g_blobs, new_cap * sizeof(unsigned char *));
    if (!new_blobs) {
        fprintf(stderr, "FATAL: out of CPU memory reallocating blob table (%d slots)\n", new_cap);
        save_to_file(1);
        exit(1);
    }
    memset(new_blobs + g_blobs_cap, 0, (new_cap - g_blobs_cap) * sizeof(unsigned char *));
    g_blobs = new_blobs;

    unsigned int *new_blob_lens = (unsigned int *)realloc(g_blob_lens, new_cap * sizeof(unsigned int));
    if (!new_blob_lens) {
        fprintf(stderr, "FATAL: out of CPU memory reallocating blob length table (%d slots)\n", new_cap);
        save_to_file(1);
        exit(1);
    }
    memset(new_blob_lens + g_blobs_cap, 0, (new_cap - g_blobs_cap) * sizeof(unsigned int));
    g_blob_lens = new_blob_lens;
    g_blobs_cap = new_cap;

    g_capacity = new_cap;
}

static void gpu_init() {
    g_capacity = INITIAL_CAP;
    CUDA_CHECK(cudaMalloc(&d_vectors, (size_t)g_capacity * g_dim * g_elem_size));
    CUDA_CHECK(cudaMalloc(&d_query, g_dim * g_elem_size));
    CUDA_CHECK(cudaMalloc(&d_dists, g_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_dists, GPU_TOP_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_ids, GPU_TOP_K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_alive, g_capacity));
    CUDA_CHECK(cudaMemset(d_alive, 1, g_capacity));

    if (g_fmt == FMT_F16) {
        d_staging_n = 1024 * g_dim;
        CUDA_CHECK(cudaMalloc(&d_staging, d_staging_n * sizeof(float)));
    }

    h_dists = (float *)malloc(g_capacity * sizeof(float));
    h_ids   = (int *)  malloc(g_capacity * sizeof(int));
    g_alive = (unsigned char *)malloc(g_capacity);
    if (!h_dists || !h_ids || !g_alive) {
        fprintf(stderr, "FATAL: out of CPU memory during init (%d slots)\n", g_capacity);
        exit(1);
    }
    memset(g_alive, 1, g_capacity);
    g_alive_cap = g_capacity;

    g_labels = (char **)calloc(g_capacity, sizeof(char *));
    if (!g_labels) {
        fprintf(stderr, "FATAL: out of CPU memory allocating label table (%d slots)\n", g_capacity);
        exit(1);
    }
    g_labels_cap = g_capacity;

    g_blobs = (unsigned char **)calloc(g_capacity, sizeof(unsigned char *));
    g_blob_lens = (unsigned int *)calloc(g_capacity, sizeof(unsigned int));
    if (!g_blobs || !g_blob_lens) {
        fprintf(stderr, "FATAL: out of CPU memory allocating blob tables (%d slots)\n", g_capacity);
        exit(1);
    }
    g_blobs_cap = g_capacity;
}

static void gpu_shutdown() {
    if (d_vectors) cudaFree(d_vectors);
    if (d_query) cudaFree(d_query);
    if (d_dists) cudaFree(d_dists);
    if (d_topk_dists) cudaFree(d_topk_dists);
    if (d_topk_ids) cudaFree(d_topk_ids);
    if (d_alive) cudaFree(d_alive);
    if (d_staging) cudaFree(d_staging);
    free(h_dists);
    free(h_ids);
    free(g_alive);
    if (g_labels) {
        for (int i = 0; i < g_labels_cap; i++) free(g_labels[i]);
        free(g_labels);
    }
    if (g_blobs) {
        for (int i = 0; i < g_blobs_cap; i++) free(g_blobs[i]);
        free(g_blobs);
    }
    free(g_blob_lens);
}

/* ===================================================================== */
/*  Vector operations                                                    */
/* ===================================================================== */

static void upload_and_store(const float *h_data, void *d_dest, int nfloats) {
    if (g_fmt == FMT_F32) {
        CUDA_CHECK(cudaMemcpy(d_dest, h_data, nfloats * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        gpu_ensure_staging(nfloats);
        CUDA_CHECK(cudaMemcpy(d_staging, h_data, nfloats * sizeof(float), cudaMemcpyHostToDevice));
        launch_f32_to_f16(d_staging, d_dest, nfloats);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

/* validate label: returns 0 ok, -1 invalid char, -2 too long, -3 empty.
 * rejected: control chars, space, and : * ? " < > | ,
 * allowed:  / \ . _ - = ! ' ^ + % & ( ) [ ] { } @ # $ ~ ` ; alphanumerics, etc. */
static int validate_label(const char *label, int len) {
    if (len <= 0) return -3;
    if (len > MAX_LABEL_BYTES) return -2;
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)label[i];
        if (c <= 0x1F || c == 0x7F) return -1;
        if (c == ' ' || c == ':' || c == '*' || c == '?' || c == '"' ||
            c == '<' || c == '>' || c == '|' || c == ',') return -1;
    }
    return 0;
}

/* set label without validation — used by .meta load (lenient, matches 1.x) */
static void vec_set_label_raw(int slot, const char *label, int len) {
    if (slot < 0 || slot >= g_labels_cap) return;
    free(g_labels[slot]);
    if (!label || len <= 0) { g_labels[slot] = NULL; return; }

    /* strip UTF-8 BOM */
    if (len >= 3 && (unsigned char)label[0] == 0xEF &&
        (unsigned char)label[1] == 0xBB && (unsigned char)label[2] == 0xBF) {
        label += 3; len -= 3;
    }
    if (len <= 0) { g_labels[slot] = NULL; return; }

    char *buf = (char *)malloc(len + 1);
    if (!buf) {
        fprintf(stderr, "ERROR: out of CPU memory storing label (len=%d), label dropped\n", len);
        g_labels[slot] = NULL;
        return;
    }
    memcpy(buf, label, len);
    buf[len] = '\0';
    g_labels[slot] = buf;
}

/* set label with strict validation — used by all wire write paths (PUSH, CMD_LABEL).
 * returns 0 on success, validate_label() error codes otherwise. on error, prior label is unchanged. */
static int vec_set_label(int slot, const char *label, int len) {
    if (slot < 0 || slot >= g_labels_cap) return -1;
    if (!label || len <= 0) {
        free(g_labels[slot]);
        g_labels[slot] = NULL;
        return 0;
    }
    int rc = validate_label(label, len);
    if (rc != 0) return rc;
    char *buf = (char *)malloc(len + 1);
    if (!buf) {
        fprintf(stderr, "ERROR: out of CPU memory storing label (len=%d), label dropped\n", len);
        return -1;
    }
    memcpy(buf, label, len);
    buf[len] = '\0';
    free(g_labels[slot]);
    g_labels[slot] = buf;
    return 0;
}

/* set blob — opaque bytes, no validation beyond length cap. len=0 clears. */
static int vec_set_blob(int slot, const unsigned char *bytes, unsigned int len) {
    if (slot < 0 || slot >= g_blobs_cap) return -1;
    if (len > MAX_DATA_BYTES) return -2;
    if (!bytes || len == 0) {
        free(g_blobs[slot]);
        g_blobs[slot] = NULL;
        g_blob_lens[slot] = 0;
        return 0;
    }
    unsigned char *buf = (unsigned char *)malloc(len);
    if (!buf) {
        fprintf(stderr, "ERROR: out of CPU memory storing blob (len=%u)\n", len);
        return -1;
    }
    memcpy(buf, bytes, len);
    free(g_blobs[slot]);
    g_blobs[slot] = buf;
    g_blob_lens[slot] = len;
    return 0;
}

static int vec_push(const float *h_vec) {
    gpu_realloc_if_needed(g_count + 1);
    int slot = g_count;
    upload_and_store(h_vec, (char *)d_vectors + (size_t)slot * g_dim * g_elem_size, g_dim);
    g_count++;
    g_dirty = 1; g_last_write = time(NULL);
    return slot;
}

static int vec_pull(const float *h_query, int *out_ids, float *out_dists, int mode) {
    int alive = g_count - g_deleted;
    if (alive <= 0) return 0;
    int n = g_count;
    int k = (alive < DEFAULT_TOP_K) ? alive : DEFAULT_TOP_K;

    cudaGetLastError();
    upload_and_store(h_query, d_query, g_dim);

    if (g_fmt == FMT_F32) {
        if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
        else launch_l2_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
    } else {
        if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
        else launch_l2_f16(d_vectors, d_query, d_dists, n, g_dim);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (n >= GPU_SORT_THRESHOLD) {
        launch_topk(d_dists, d_alive, n, d_topk_dists, d_topk_ids);
        CUDA_CHECK(cudaMemcpy(h_dists, d_topk_dists, k * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ids, d_topk_ids, k * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(h_dists, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));
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
    }

    for (int i = 0; i < k; i++) {
        out_ids[i] = h_ids[i];
        out_dists[i] = h_dists[i];
    }
    return k;
}

/* same as vec_pull but the query is an existing slot — copies device-to-device */
static int vec_pull_by_idx(int idx, int *out_ids, float *out_dists, int mode) {
    int alive = g_count - g_deleted;
    if (alive <= 0) return 0;
    int n = g_count;
    int k = (alive < DEFAULT_TOP_K) ? alive : DEFAULT_TOP_K;

    cudaGetLastError();
    CUDA_CHECK(cudaMemcpy(d_query, (char *)d_vectors + (size_t)idx * g_dim * g_elem_size,
                          g_dim * g_elem_size, cudaMemcpyDeviceToDevice));

    if (g_fmt == FMT_F32) {
        if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
        else launch_l2_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
    } else {
        if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
        else launch_l2_f16(d_vectors, d_query, d_dists, n, g_dim);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (n >= GPU_SORT_THRESHOLD) {
        launch_topk(d_dists, d_alive, n, d_topk_dists, d_topk_ids);
        CUDA_CHECK(cudaMemcpy(h_dists, d_topk_dists, k * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_ids, d_topk_ids, k * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(h_dists, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));
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
    }

    for (int i = 0; i < k; i++) {
        out_ids[i] = h_ids[i];
        out_dists[i] = h_dists[i];
    }
    return k;
}

/* ===================================================================== */
/*  Clustering (DBSCAN)                                                  */
/* ===================================================================== */

typedef int (*write_fn)(void *ctx, const char *buf, int len);

#define CLUSTER_UNVISITED -1
#define CLUSTER_NOISE     -2

/*
 * vec_cluster: DBSCAN clustering using existing GPU distance kernels.
 * Each seed does a device-to-device copy into d_query, launches the distance
 * kernel, copies distances back, and thresholds to find neighbors.
 * No new CUDA kernels needed.
 *
 * Returns number of clusters found. Writes results via writer callback.
 * Format: one line per cluster "id:member,member,...\n", noise last.
 */
static void vec_cluster(float eps, int min_pts, int mode, write_fn writer, void *wctx) {
    int n = g_count;
    int alive = n - g_deleted;
    if (alive <= 0) {
        writer(wctx, "end\n", 4);
        return;
    }

    long long t_start = now_ms();
    fprintf(stderr, "cluster: %d vectors, eps=%.4g, min_pts=%d, %s\n",
            alive, eps, min_pts, mode == 1 ? "cosine" : "L2");

    float eps_sq = (mode == 1) ? eps : eps * eps; /* cosine is already 0..2, L2 we store squared */

    int *cluster_id = (int *)malloc(n * sizeof(int));
    int *queue      = (int *)malloc(n * sizeof(int));
    float *dists_buf = (float *)malloc(n * sizeof(float));
    if (!cluster_id || !queue || !dists_buf) {
        fprintf(stderr, "cluster: out of CPU memory (n=%d)\n", n);
        free(cluster_id); free(queue); free(dists_buf);
        writer(wctx, "err out of memory\n", 18);
        return;
    }
    for (int i = 0; i < n; i++)
        cluster_id[i] = g_alive[i] ? CLUSTER_UNVISITED : CLUSTER_NOISE;

    int cluster = 0;

    for (int i = 0; i < n; i++) {
        if (cluster_id[i] != CLUSTER_UNVISITED) continue;

        /* compute distances from vector i to all others using GPU */
        CUDA_CHECK(cudaMemcpy(d_query, (char *)d_vectors + (size_t)i * g_dim * g_elem_size,
                              g_dim * g_elem_size, cudaMemcpyDeviceToDevice));

        if (g_fmt == FMT_F32) {
            if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
            else launch_l2_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
        } else {
            if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
            else launch_l2_f16(d_vectors, d_query, d_dists, n, g_dim);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dists_buf, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));

        /* find neighbors within eps */
        int q_head = 0, q_tail = 0;
        int neighbor_count = 0;
        for (int j = 0; j < n; j++) {
            if (!g_alive[j]) continue;
            if (dists_buf[j] <= eps_sq) neighbor_count++;
        }

        if (neighbor_count < min_pts) {
            cluster_id[i] = CLUSTER_NOISE;
            continue;
        }

        /* start new cluster — assign seed and enqueue all initial neighbors */
        cluster_id[i] = cluster;
        for (int j = 0; j < n; j++) {
            if (!g_alive[j] || dists_buf[j] > eps_sq) continue;
            if (cluster_id[j] == CLUSTER_NOISE) {
                cluster_id[j] = cluster;
            } else if (cluster_id[j] == CLUSTER_UNVISITED) {
                cluster_id[j] = cluster; /* mark immediately to prevent duplicates */
                queue[q_tail++] = j;
            }
        }

        /* process queue — expand cluster */
        while (q_head < q_tail) {
            int j = queue[q_head++];

            /* range query from j */
            CUDA_CHECK(cudaMemcpy(d_query, (char *)d_vectors + (size_t)j * g_dim * g_elem_size,
                                  g_dim * g_elem_size, cudaMemcpyDeviceToDevice));
            if (g_fmt == FMT_F32) {
                if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
                else launch_l2_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
            } else {
                if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
                else launch_l2_f16(d_vectors, d_query, d_dists, n, g_dim);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(dists_buf, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));

            int j_neighbors = 0;
            for (int k = 0; k < n; k++) {
                if (!g_alive[k]) continue;
                if (dists_buf[k] <= eps_sq) j_neighbors++;
            }
            if (j_neighbors >= min_pts) {
                /* add unvisited neighbors to queue; mark immediately to prevent duplicates */
                for (int k = 0; k < n; k++) {
                    if (!g_alive[k]) continue;
                    if (dists_buf[k] <= eps_sq) {
                        if (cluster_id[k] == CLUSTER_NOISE) {
                            cluster_id[k] = cluster;
                        } else if (cluster_id[k] == CLUSTER_UNVISITED) {
                            cluster_id[k] = cluster; /* mark before enqueue — prevents duplicates */
                            queue[q_tail++] = k;
                        }
                    }
                }
            }
        }
        cluster++;
    }

    /* output results */
    char line[256];
    int line_len;

    /* one line per cluster: member,member,...\n */
    for (int c = 0; c < cluster; c++) {
        int first = 1;
        for (int i = 0; i < n; i++) {
            if (cluster_id[i] != c) continue;
            const char *lbl = (i < g_labels_cap) ? g_labels[i] : NULL;
            if (lbl)
                line_len = snprintf(line, sizeof(line), "%s%s", first ? "" : ",", lbl);
            else
                line_len = snprintf(line, sizeof(line), "%s%d", first ? "" : ",", i);
            writer(wctx, line, line_len);
            first = 0;
        }
        writer(wctx, "\n", 1);
    }

    /* noise: same format, one line */
    {
        int first = 1;
        int has_noise = 0;
        for (int i = 0; i < n; i++) {
            if (cluster_id[i] != CLUSTER_NOISE || !g_alive[i]) continue;
            has_noise = 1;
            const char *lbl = (i < g_labels_cap) ? g_labels[i] : NULL;
            if (lbl)
                line_len = snprintf(line, sizeof(line), "%s%s", first ? "" : ",", lbl);
            else
                line_len = snprintf(line, sizeof(line), "%s%d", first ? "" : ",", i);
            writer(wctx, line, line_len);
            first = 0;
        }
        if (has_noise) writer(wctx, "\n", 1);
    }

    line_len = snprintf(line, sizeof(line), "end\n");
    writer(wctx, line, line_len);

    /* count noise for logging */
    int noise_count = 0;
    for (int i = 0; i < n; i++)
        if (cluster_id[i] == CLUSTER_NOISE && g_alive[i]) noise_count++;
    char el[32]; format_elapsed(now_ms() - t_start, el, sizeof(el));
    fprintf(stderr, "cluster: %d clusters, %d noise (%s)\n", cluster, noise_count, el);

    free(cluster_id);
    free(queue);
    free(dists_buf);
}

/* ===================================================================== */
/*  Represent: one most-distinct member per DBSCAN cluster              */
/* ===================================================================== */

/*
 * vec_represent: runs DBSCAN internally, then for each cluster picks the
 * member farthest from that cluster's centroid — the most atypical, boundary-
 * sitting face rather than the average one. Noise points are skipped.
 * Output: one slot index or label per line, then "end\n".
 */
static void vec_represent(float eps, int min_pts, int mode, write_fn writer, void *wctx) {
    int n = g_count;
    int alive = n - g_deleted;
    char resp[256];
    int rlen;

    if (alive <= 0) { writer(wctx, "end\n", 4); return; }

    long long t_start = now_ms();
    fprintf(stderr, "represent: %d vectors, eps=%.4g, min_pts=%d, %s\n",
            alive, eps, min_pts, mode == 1 ? "cosine" : "L2");

    /* ---- phase 1: DBSCAN (identical to vec_cluster) ---- */
    int   *cluster_id = (int *)  malloc(n * sizeof(int));
    int   *queue      = (int *)  malloc(n * sizeof(int));
    float *dists_buf  = (float *)malloc(n * sizeof(float));
    if (!cluster_id || !queue || !dists_buf) {
        fprintf(stderr, "represent: out of CPU memory (n=%d)\n", n);
        free(cluster_id); free(queue); free(dists_buf);
        writer(wctx, "err out of memory\n", 18);
        return;
    }

    for (int i = 0; i < n; i++)
        cluster_id[i] = g_alive[i] ? CLUSTER_UNVISITED : CLUSTER_NOISE;

    float eps_sq = (mode == 1) ? eps : eps * eps;
    int num_clusters = 0;

    for (int i = 0; i < n; i++) {
        if (cluster_id[i] != CLUSTER_UNVISITED) continue;

        CUDA_CHECK(cudaMemcpy(d_query,
            (char *)d_vectors + (size_t)i * g_dim * g_elem_size,
            g_dim * g_elem_size, cudaMemcpyDeviceToDevice));

        if (g_fmt == FMT_F32) {
            if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
            else           launch_l2_f32 ((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
        } else {
            if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
            else           launch_l2_f16 (d_vectors, d_query, d_dists, n, g_dim);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dists_buf, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));

        int q_head = 0, q_tail = 0, neighbor_count = 0;
        for (int j = 0; j < n; j++) {
            if (!g_alive[j]) continue;
            if (dists_buf[j] <= eps_sq) neighbor_count++;
        }
        if (neighbor_count < min_pts) { cluster_id[i] = CLUSTER_NOISE; continue; }

        /* assign seed and enqueue initial neighbors, marking immediately */
        cluster_id[i] = num_clusters;
        for (int j = 0; j < n; j++) {
            if (!g_alive[j] || dists_buf[j] > eps_sq) continue;
            if (cluster_id[j] == CLUSTER_NOISE) {
                cluster_id[j] = num_clusters;
            } else if (cluster_id[j] == CLUSTER_UNVISITED) {
                cluster_id[j] = num_clusters;
                queue[q_tail++] = j;
            }
        }

        while (q_head < q_tail) {
            int j = queue[q_head++];

            CUDA_CHECK(cudaMemcpy(d_query,
                (char *)d_vectors + (size_t)j * g_dim * g_elem_size,
                g_dim * g_elem_size, cudaMemcpyDeviceToDevice));
            if (g_fmt == FMT_F32) {
                if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
                else           launch_l2_f32 ((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
            } else {
                if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
                else           launch_l2_f16 (d_vectors, d_query, d_dists, n, g_dim);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(dists_buf, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));

            int j_neighbors = 0;
            for (int k = 0; k < n; k++) {
                if (!g_alive[k]) continue;
                if (dists_buf[k] <= eps_sq) j_neighbors++;
            }
            if (j_neighbors >= min_pts) {
                for (int k = 0; k < n; k++) {
                    if (!g_alive[k]) continue;
                    if (dists_buf[k] <= eps_sq) {
                        if (cluster_id[k] == CLUSTER_NOISE) {
                            cluster_id[k] = num_clusters;
                        } else if (cluster_id[k] == CLUSTER_UNVISITED) {
                            cluster_id[k] = num_clusters;
                            queue[q_tail++] = k;
                        }
                    }
                }
            }
        }
        num_clusters++;
    }

    if (num_clusters == 0) {
        char el[32]; format_elapsed(now_ms() - t_start, el, sizeof(el));
        fprintf(stderr, "represent: 0 clusters found (%s)\n", el);
        writer(wctx, "end\n", 4);
        free(cluster_id); free(queue); free(dists_buf);
        return;
    }

    fprintf(stderr, "represent: %d clusters found, picking representatives\n", num_clusters);

    /* ---- phase 2: download all vectors to CPU for centroid math ---- */
    size_t total_bytes = (size_t)n * g_dim * g_elem_size;
    float *all_vecs_f32 = (float *)malloc((size_t)n * g_dim * sizeof(float));
    if (!all_vecs_f32) {
        fprintf(stderr, "ERROR: out of CPU memory downloading vectors for represent (n=%d)\n", n);
        free(cluster_id); free(queue); free(dists_buf);
        writer(wctx, "err out of memory\n", 18);
        return;
    }

    if (g_fmt == FMT_F32) {
        CUDA_CHECK(cudaMemcpy(all_vecs_f32, d_vectors, total_bytes, cudaMemcpyDeviceToHost));
    } else {
        /* f16: download raw, convert to f32 on CPU */
        unsigned short *raw = (unsigned short *)malloc(total_bytes);
        if (!raw) {
            fprintf(stderr, "ERROR: out of CPU memory for f16 conversion in represent\n");
            free(all_vecs_f32); free(cluster_id); free(queue); free(dists_buf);
            writer(wctx, "err out of memory\n", 18);
            return;
        }
        CUDA_CHECK(cudaMemcpy(raw, d_vectors, total_bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n * g_dim; i++) {
            unsigned int bits = raw[i];
            unsigned int sign = (bits >> 15) & 1;
            unsigned int exp  = (bits >> 10) & 0x1F;
            unsigned int mant = bits & 0x3FF;
            unsigned int f32;
            if      (exp == 0)    f32 = sign << 31;
            else if (exp == 31)   f32 = (sign << 31) | 0x7F800000 | (mant << 13);
            else                  f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&all_vecs_f32[i], &f32, sizeof(float));
        }
        free(raw);
    }

    /* ---- phase 3: compute centroid per cluster, find farthest member ---- */
    float *centroid = (float *)malloc(g_dim * sizeof(float));
    int   *result   = (int *)  malloc(num_clusters * sizeof(int));
    if (!centroid || !result) {
        fprintf(stderr, "ERROR: out of CPU memory for centroids in represent\n");
        free(centroid); free(result); free(all_vecs_f32);
        free(cluster_id); free(queue); free(dists_buf);
        writer(wctx, "err out of memory\n", 18);
        return;
    }

    for (int c = 0; c < num_clusters; c++) {
        /* compute centroid */
        memset(centroid, 0, g_dim * sizeof(float));
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (cluster_id[i] != c) continue;
            float *v = all_vecs_f32 + (size_t)i * g_dim;
            for (int d = 0; d < g_dim; d++) centroid[d] += v[d];
            count++;
        }
        if (count == 0) { result[c] = -1; continue; }
        for (int d = 0; d < g_dim; d++) centroid[d] /= (float)count;

        /* upload centroid as query, run distance kernel against full DB */
        CUDA_CHECK(cudaMemcpy(d_query, centroid, g_dim * sizeof(float), cudaMemcpyHostToDevice));
        if (g_fmt == FMT_F32) {
            if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
            else           launch_l2_f32 ((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
        } else {
            if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
            else           launch_l2_f16 (d_vectors, d_query, d_dists, n, g_dim);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dists_buf, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));

        /* pick farthest cluster member from centroid */
        int   best      = -1;
        float best_dist = -1.0f;
        for (int i = 0; i < n; i++) {
            if (cluster_id[i] != c) continue;
            if (dists_buf[i] > best_dist) { best_dist = dists_buf[i]; best = i; }
        }
        result[c] = best;
    }

    /* ---- output ---- */
    char line[256];
    int  line_len;
    for (int c = 0; c < num_clusters; c++) {
        int idx = result[c];
        if (idx < 0) continue;
        const char *lbl = (idx < g_labels_cap) ? g_labels[idx] : NULL;
        if (lbl) line_len = snprintf(line, sizeof(line), "%s\n", lbl);
        else     line_len = snprintf(line, sizeof(line), "%d\n", idx);
        writer(wctx, line, line_len);
    }
    writer(wctx, "end\n", 4);
    {
        char el[32]; format_elapsed(now_ms() - t_start, el, sizeof(el));
        fprintf(stderr, "represent: done, %d representatives (%s)\n", num_clusters, el);
    }

    free(centroid); free(result); free(all_vecs_f32);
    free(cluster_id); free(queue); free(dists_buf);
}

/* ===================================================================== */
/*  Farthest Point Sampling                                              */
/* ===================================================================== */

/*
 * vec_distinct: selects k maximally spread-out vectors using greedy FPS.
 * Each iteration picks the vector farthest from any already-selected point.
 * Reuses existing GPU distance kernels — no new CUDA code needed.
 * Output: one slot index (or label) per line, then "end\n".
 */
static void vec_distinct(int k, int mode, write_fn writer, void *wctx) {
    int n = g_count;
    int alive = n - g_deleted;
    char resp[256];
    int rlen;

    if (alive <= 0) { writer(wctx, "end\n", 4); return; }
    if (k <= 0) {
        rlen = snprintf(resp, sizeof(resp), "err k must be > 0\n");
        writer(wctx, resp, rlen); return;
    }
    if (k > alive) k = alive; /* clamp — can't pick more than we have */

    long long t_start = now_ms();
    fprintf(stderr, "distinct: selecting %d from %d vectors, %s\n",
            k, alive, mode == 1 ? "cosine" : "L2");

    float   *min_dist  = (float *)malloc(n * sizeof(float));
    int     *selected  = (int *)  malloc(k * sizeof(int));
    float   *dists_buf = (float *)malloc(n * sizeof(float));
    if (!min_dist || !selected || !dists_buf) {
        fprintf(stderr, "ERROR: out of CPU memory for distinct (n=%d, k=%d)\n", n, k);
        free(min_dist); free(selected); free(dists_buf);
        writer(wctx, "err out of memory\n", 18);
        return;
    }

    /* initialise min_dist: FLT_MAX for alive, -1 for dead (never picked) */
    for (int i = 0; i < n; i++)
        min_dist[i] = g_alive[i] ? FLT_MAX : -1.0f;

    /* seed: first alive vector */
    int seed = -1;
    for (int i = 0; i < n; i++) { if (g_alive[i]) { seed = i; break; } }
    selected[0] = seed;
    min_dist[seed] = -1.0f; /* mark as selected */

    for (int step = 1; step < k; step++) {
        int last = selected[step - 1];

        /* upload last selected vector as query */
        CUDA_CHECK(cudaMemcpy(d_query,
            (char *)d_vectors + (size_t)last * g_dim * g_elem_size,
            g_dim * g_elem_size, cudaMemcpyDeviceToDevice));

        if (g_fmt == FMT_F32) {
            if (mode == 1) launch_cos_f32((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
            else           launch_l2_f32 ((const float *)d_vectors, (const float *)d_query, d_dists, n, g_dim);
        } else {
            if (mode == 1) launch_cos_f16(d_vectors, d_query, d_dists, n, g_dim);
            else           launch_l2_f16 (d_vectors, d_query, d_dists, n, g_dim);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dists_buf, d_dists, n * sizeof(float), cudaMemcpyDeviceToHost));

        /* update min_dist and find next farthest */
        int   best     = -1;
        float best_dist = -1.0f;
        for (int i = 0; i < n; i++) {
            if (min_dist[i] < 0.0f) continue; /* dead or already selected */
            if (dists_buf[i] < min_dist[i]) min_dist[i] = dists_buf[i];
            if (min_dist[i] > best_dist) { best_dist = min_dist[i]; best = i; }
        }

        if (best < 0) break; /* exhausted alive vectors (k was clamped but guard anyway) */
        selected[step] = best;
        min_dist[best] = -1.0f; /* mark selected */
    }

    /* output */
    char line[256];
    int  line_len;
    for (int s = 0; s < k; s++) {
        int idx = selected[s];
        const char *lbl = (idx < g_labels_cap) ? g_labels[idx] : NULL;
        if (lbl) line_len = snprintf(line, sizeof(line), "%s\n", lbl);
        else     line_len = snprintf(line, sizeof(line), "%d\n", idx);
        writer(wctx, line, line_len);
    }
    writer(wctx, "end\n", 4);
    {
        char el[32]; format_elapsed(now_ms() - t_start, el, sizeof(el));
        fprintf(stderr, "distinct: done, %d selected (%s)\n", k, el);
    }

    free(min_dist);
    free(selected);
    free(dists_buf);
}

/* ===================================================================== */
/*  Persistence                                                          */
/* ===================================================================== */

static unsigned int g_loaded_crc = 0;
static unsigned int g_computed_crc = 0;
static int g_crc_ok = 0;

static long long estimate_file_size() {
    long long tensors = 13LL + g_count + (long long)g_count * g_dim * g_elem_size + 4;
    long long meta = 4; /* count header */
    for (int i = 0; i < g_count; i++)
        meta += 4 + (g_labels[i] ? (long long)strlen(g_labels[i]) : 0);
    return tensors + meta;
}

static void save_to_file(int already_locked) {
    if (g_count == 0) return;
    if (g_readonly) return;
    if (!g_dirty && g_crc_ok == 1) return;

    long long t_save_start = now_ms();

    /* disk space check */
    long long needed = estimate_file_size() + (1024 * 1024);
#ifdef _WIN32
    ULARGE_INTEGER free_bytes;
    if (GetDiskFreeSpaceExA(NULL, &free_bytes, NULL, NULL)) {
        if ((long long)free_bytes.QuadPart < needed) {
            fprintf(stderr, "ERROR: not enough disk space (need %lld bytes, have %llu)\n",
                    needed, (unsigned long long)free_bytes.QuadPart);
            return;
        }
    }
#else
    struct statvfs st;
    if (statvfs(".", &st) == 0) {
        long long avail = (long long)st.f_bavail * st.f_frsize;
        if (avail < needed) {
            fprintf(stderr, "ERROR: not enough disk space (need %lld bytes, have %lld)\n",
                    needed, avail);
            return;
        }
    }
#endif

    /* --- snapshot under lock, write to disk after --- */
    int snap_count   = g_count;
    int snap_deleted = g_deleted;
    int snap_dim     = g_dim;
    int snap_fmt     = g_fmt;
    int snap_elem    = g_elem_size;

    unsigned char *snap_alive = (unsigned char *)malloc(snap_count);
    if (!snap_alive) {
        fprintf(stderr, "ERROR: out of CPU memory snapshotting alive mask (%d bytes), save aborted\n", snap_count);
        return;
    }
    memcpy(snap_alive, g_alive, snap_count);

    size_t total_bytes = (size_t)snap_count * snap_dim * snap_elem;
    void *h_buf = NULL;
    if (total_bytes > 0) {
        h_buf = malloc(total_bytes);
        if (!h_buf) {
            fprintf(stderr, "ERROR: out of CPU memory during save (%zu bytes), save aborted\n", total_bytes);
            free(snap_alive);
            return;
        }
        CUDA_CHECK(cudaMemcpy(h_buf, d_vectors, total_bytes, cudaMemcpyDeviceToHost));
    }

    /* snapshot complete — mark clean and release lock so clients aren't blocked during disk I/O */
    g_dirty = 0;
    if (already_locked) REQ_UNLOCK();

    FILE *f = fopen(g_filepath, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for writing\n", g_filepath);
        free(h_buf); free(snap_alive);
        if (already_locked) REQ_LOCK();
        return;
    }

    fwrite(&snap_dim,     sizeof(int), 1, f);
    fwrite(&snap_count,   sizeof(int), 1, f);
    fwrite(&snap_deleted, sizeof(int), 1, f);
    unsigned char fmt_byte = (unsigned char)snap_fmt;
    fwrite(&fmt_byte, 1, 1, f);

    unsigned int crc = 0;
    if (total_bytes > 0) {
        fwrite(snap_alive, 1, snap_count, f);
        crc = crc32_update(crc, snap_alive, snap_count);
        fwrite(h_buf, 1, total_bytes, f);
        crc = crc32_update(crc, h_buf, total_bytes);
        free(h_buf);
    }
    free(snap_alive);

    fwrite(&crc, sizeof(unsigned int), 1, f);
    fclose(f);

    char crc_name[16];
    crc32_word(crc, crc_name);
    g_crc_ok = 1;
    g_loaded_crc = crc;
    g_computed_crc = crc;
    {
        char el[32]; format_elapsed(now_ms() - t_save_start, el, sizeof(el));
        printf("saved %d vectors to %s [%s 0x%08X] (%s)\n", snap_count, g_filepath, crc_name, crc, el);
    }

    if (already_locked) REQ_LOCK(); /* reacquire so caller's unlock/paired call stays balanced */

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

    /* save data blobs to .data sidecar */
    int has_blobs = 0;
    for (int i = 0; i < g_count; i++) { if (g_blobs[i] && g_blob_lens[i] > 0) { has_blobs = 1; break; } }
    if (has_blobs) {
        char datapath[512];
        strncpy(datapath, g_filepath, sizeof(datapath) - 1);
        char *ext = strstr(datapath, ".tensors");
        if (ext) strcpy(ext, ".data");
        else snprintf(datapath, sizeof(datapath), "%s.data", g_name);

        FILE *df = fopen(datapath, "wb");
        if (df) {
            unsigned char *mask = (unsigned char *)malloc(g_count);
            if (mask) {
                for (int i = 0; i < g_count; i++) {
                    mask[i] = (g_blobs[i] && g_blob_lens[i] > 0) ? 1 : 0;
                }
                fwrite(&g_count, sizeof(int), 1, df);
                fwrite(mask, 1, g_count, df);
                unsigned int dcrc = crc32_update(0, mask, g_count);
                for (int i = 0; i < g_count; i++) {
                    if (!mask[i]) continue;
                    unsigned int dlen = g_blob_lens[i];
                    fwrite(&dlen, sizeof(unsigned int), 1, df);
                    fwrite(g_blobs[i], 1, dlen, df);
                    dcrc = crc32_update(dcrc, (const unsigned char *)&dlen, sizeof(unsigned int));
                    dcrc = crc32_update(dcrc, g_blobs[i], dlen);
                }
                fwrite(&dcrc, sizeof(unsigned int), 1, df);
                free(mask);
            }
            fclose(df);
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

/* g_loaded_crc, g_computed_crc, g_crc_ok moved before save_to_file */

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
        fprintf(stderr, "WARN: %s is empty or corrupt, removing and starting fresh\n", g_filepath);
        remove(g_filepath);
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
        if (!h_buf) {
            fprintf(stderr, "FATAL: out of CPU memory loading vectors (%zu bytes)\n", total_bytes);
            fclose(f);
            return 0;
        }
        size_t rd = fread(h_buf, 1, total_bytes, f);
        if (rd != total_bytes) {
            fprintf(stderr, "WARN: data truncated\n");
            file_count = (int)(rd / (g_dim * g_elem_size));
        }
        crc = crc32_update(crc, h_buf, rd);
        CUDA_CHECK(cudaMemcpy(d_vectors, h_buf, (size_t)file_count * g_dim * g_elem_size, cudaMemcpyHostToDevice));
        free(h_buf);
        CUDA_CHECK(cudaMemcpy(d_alive, g_alive, file_count, cudaMemcpyHostToDevice));
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
                    if (!g_labels[i]) { fprintf(stderr, "WARN: out of memory loading label[%d], skipping\n", i); break; }
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

    /* load data blobs from .data sidecar (lenient: skip on any corruption) */
    char datapath[512];
    strncpy(datapath, g_filepath, sizeof(datapath) - 1);
    char *dext = strstr(datapath, ".tensors");
    if (dext) strcpy(dext, ".data");
    else snprintf(datapath, sizeof(datapath), "%s.data", g_name);

    FILE *df = fopen(datapath, "rb");
    if (df) {
        int data_count = 0;
        if (fread(&data_count, sizeof(int), 1, df) == 1 && data_count > 0 && data_count <= g_count) {
            unsigned char *mask = (unsigned char *)malloc(data_count);
            if (mask && fread(mask, 1, data_count, df) == (size_t)data_count) {
                for (int i = 0; i < data_count; i++) {
                    if (!mask[i]) continue;
                    unsigned int dlen = 0;
                    if (fread(&dlen, sizeof(unsigned int), 1, df) != 1) break;
                    if (dlen == 0 || dlen > MAX_DATA_BYTES) break;
                    unsigned char *buf = (unsigned char *)malloc(dlen);
                    if (!buf) { fprintf(stderr, "WARN: out of memory loading blob[%d], stopping\n", i); break; }
                    if (fread(buf, 1, dlen, df) != (size_t)dlen) { free(buf); break; }
                    g_blobs[i] = buf;
                    g_blob_lens[i] = dlen;
                }
            }
            free(mask);
        }
        fclose(df);
    }

    return 1;
}

/* ===================================================================== */
/*  Protocol parser                                                      */
/* ===================================================================== */

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

/* ===================================================================== */
/*  Binary frame protocol                                                */
/* ===================================================================== */

/* VEC 2.0: body_len is explicit (4B u32 after label) — no per-cmd inference.
 * Returns body_len, -1 if not enough bytes to read the length field, -2 if unknown command. */
static int frame_data_len(unsigned char cmd, const char *data_start, int available, int label_len) {
    (void)label_len;
    switch (cmd) {
        case CMD_PUSH:
        case CMD_QUERY:
        case CMD_GET:
        case CMD_UPDATE:
        case CMD_DELETE:
        case CMD_LABEL:
        case CMD_UNDO:
        case CMD_SAVE:
        case CMD_CLUSTER:
        case CMD_DISTINCT:
        case CMD_REPRESENT:
        case CMD_INFO:
        case CMD_QID:
        case CMD_SET_DATA:
        case CMD_GET_DATA:
            if (available < 4) return -1;
            unsigned int blen;
            memcpy(&blen, data_start, 4);
            return (int)blen;
        default:
            return -2;
    }
}

/* ===================================================================== */
/*  2.0 response envelope helpers                                        */
/* ===================================================================== */

/* write a binary OK response: [01 status][4B len=body_len][body...] */
static void resp_ok_header(write_fn writer, void *wctx, unsigned int body_len) {
    char hdr[5];
    hdr[0] = (char)RESP_OK;
    memcpy(hdr + 1, &body_len, 4);
    writer(wctx, hdr, 5);
}

static void resp_ok_empty(write_fn writer, void *wctx) {
    char hdr[5] = { (char)RESP_OK, 0, 0, 0, 0 };
    writer(wctx, hdr, 5);
}

static void resp_err(write_fn writer, void *wctx, const char *msg) {
    unsigned int el = (unsigned int)strlen(msg);
    char hdr[5];
    hdr[0] = (char)RESP_ERR;
    memcpy(hdr + 1, &el, 4);
    writer(wctx, hdr, 5);
    if (el > 0) writer(wctx, msg, (int)el);
}

/* persistent buffers for bin_write_vec — avoids malloc per call */
static float *g_bwv_f32 = NULL;
static unsigned short *g_bwv_f16 = NULL;
static int g_bwv_dim = 0;

static void bin_write_vec_ensure(int dim) {
    if (dim <= g_bwv_dim) return;
    free(g_bwv_f32); free(g_bwv_f16);
    g_bwv_f32 = (float *)malloc(dim * sizeof(float));
    g_bwv_f16 = (unsigned short *)malloc(dim * sizeof(unsigned short));
    g_bwv_dim = dim;
}

/* write one vector from GPU to client as raw fp32 (handles f16 decode) */
static int bin_write_vec(int idx, write_fn writer, void *wctx) {
    bin_write_vec_ensure(g_dim);
    if (!g_bwv_f32) { fprintf(stderr, "ERROR: out of CPU memory in bin_write_vec (dim=%d)\n", g_dim); return -1; }
    if (g_fmt == FMT_F32) {
        CUDA_CHECK(cudaMemcpy(g_bwv_f32, (char *)d_vectors + (size_t)idx * g_dim * g_elem_size, g_dim * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(g_bwv_f16, (char *)d_vectors + (size_t)idx * g_dim * g_elem_size, g_dim * sizeof(unsigned short), cudaMemcpyDeviceToHost));
        for (int d = 0; d < g_dim; d++) {
            unsigned int bits = g_bwv_f16[d];
            unsigned int sign = (bits >> 15) & 1;
            unsigned int exp = (bits >> 10) & 0x1F;
            unsigned int mant = bits & 0x3FF;
            unsigned int f32;
            if (exp == 0) f32 = sign << 31;
            else if (exp == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13);
            else f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&g_bwv_f32[d], &f32, sizeof(float));
        }
    }
    writer(wctx, (const char *)g_bwv_f32, g_dim * (int)sizeof(float));
    return 0;
}

/* resolve label to a single index, -1 if not found, -2 if ambiguous */
static int resolve_label(const char *label, int label_len) {
    int idx = -1;
    for (int i = 0; i < g_count; i++) {
        if (!g_alive[i] || !g_labels[i]) continue;
        if ((int)strlen(g_labels[i]) == label_len && strncmp(g_labels[i], label, label_len) == 0) {
            if (idx >= 0) return -2;
            idx = i;
        }
    }
    return idx;
}

/* resolve label to all matching indices, returns count */
static int resolve_label_all(const char *label, int label_len, int **out_ids) {
    int *ids = NULL;
    int count = 0;
    for (int i = 0; i < g_count; i++) {
        if (!g_alive[i] || !g_labels[i]) continue;
        if ((int)strlen(g_labels[i]) == label_len && strncmp(g_labels[i], label, label_len) == 0) {
            int *tmp = (int *)realloc(ids, (count + 1) * sizeof(int));
            if (!tmp) { free(ids); *out_ids = NULL; return 0; }
            ids = tmp;
            ids[count++] = i;
        }
    }
    *out_ids = ids;
    return count;
}

/* scratch writer — collects writer_fn output into a growable buffer.
 * used to wrap legacy text-emitting cluster/distinct/represent into binary envelope. */
typedef struct {
    char *buf;
    unsigned int len;
    unsigned int cap;
    int oom;
} scratch_writer_ctx;

static void scratch_writer_init(scratch_writer_ctx *s) {
    s->buf = NULL; s->len = 0; s->cap = 0; s->oom = 0;
}
static void scratch_writer_free(scratch_writer_ctx *s) {
    free(s->buf); s->buf = NULL; s->len = 0; s->cap = 0;
}
static int scratch_writer_fn(void *ctx, const char *data, int n) {
    scratch_writer_ctx *s = (scratch_writer_ctx *)ctx;
    if (s->oom || n <= 0) return n;
    unsigned int need = s->len + (unsigned int)n;
    if (need > s->cap) {
        unsigned int newcap = s->cap ? s->cap * 2 : 4096;
        while (newcap < need) newcap *= 2;
        char *nb = (char *)realloc(s->buf, newcap);
        if (!nb) { s->oom = 1; return n; }
        s->buf = nb; s->cap = newcap;
    }
    memcpy(s->buf + s->len, data, n);
    s->len += (unsigned int)n;
    return n;
}

/* read a host-fp32 vector from device into a caller-provided buffer (decodes f16) */
static int read_vec_from_device(int idx, float *out_dim_f32) {
    bin_write_vec_ensure(g_dim);
    if (!g_bwv_f32) return -1;
    if (g_fmt == FMT_F32) {
        CUDA_CHECK(cudaMemcpy(out_dim_f32, (char *)d_vectors + (size_t)idx * g_dim * g_elem_size, g_dim * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(g_bwv_f16, (char *)d_vectors + (size_t)idx * g_dim * g_elem_size, g_dim * sizeof(unsigned short), cudaMemcpyDeviceToHost));
        for (int d = 0; d < g_dim; d++) {
            unsigned int bits = g_bwv_f16[d];
            unsigned int sign = (bits >> 15) & 1;
            unsigned int exp = (bits >> 10) & 0x1F;
            unsigned int mant = bits & 0x3FF;
            unsigned int f32;
            if (exp == 0) f32 = sign << 31;
            else if (exp == 31) f32 = (sign << 31) | 0x7F800000 | (mant << 13);
            else f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&out_dim_f32[d], &f32, sizeof(float));
        }
    }
    return 0;
}

/* compute body length for a record under a shape mask (excludes leading 4B index, includes optional distance via with_distance flag) */
static unsigned int record_body_len(int idx, unsigned char shape, int with_distance) {
    unsigned int n = 4; /* index */
    if (with_distance) n += 4;
    if (shape & SHAPE_LABEL) {
        unsigned int ll = (idx >= 0 && idx < g_labels_cap && g_labels[idx]) ? (unsigned int)strlen(g_labels[idx]) : 0;
        n += 4 + ll;
    }
    if (shape & SHAPE_DATA) {
        unsigned int dl = (idx >= 0 && idx < g_blobs_cap && g_blobs[idx]) ? g_blob_lens[idx] : 0;
        n += 4 + dl;
    }
    if (shape & SHAPE_VECTOR) n += g_dim * 4;
    return n;
}

/* build a result body containing count records under shape mask; returns malloc'd buffer + length via *out_len.
 * if with_distance, also writes 4B distance after each index using dists[]. */
static char *build_result_body(const int *ids, const float *dists, int count, unsigned char shape,
                               int with_distance, unsigned int *out_len) {
    unsigned int total = 4; /* count prefix */
    for (int i = 0; i < count; i++) total += record_body_len(ids[i], shape, with_distance);

    char *buf = (char *)malloc(total);
    if (!buf) return NULL;
    char *p = buf;
    unsigned int u_count = (unsigned int)count;
    memcpy(p, &u_count, 4); p += 4;

    for (int i = 0; i < count; i++) {
        int idx = ids[i];
        memcpy(p, &idx, 4); p += 4;
        if (with_distance) {
            float d = dists[i];
            memcpy(p, &d, 4); p += 4;
        }
        if (shape & SHAPE_LABEL) {
            unsigned int ll = (idx >= 0 && idx < g_labels_cap && g_labels[idx]) ? (unsigned int)strlen(g_labels[idx]) : 0;
            memcpy(p, &ll, 4); p += 4;
            if (ll > 0) { memcpy(p, g_labels[idx], ll); p += ll; }
        }
        if (shape & SHAPE_DATA) {
            unsigned int dl = (idx >= 0 && idx < g_blobs_cap && g_blobs[idx]) ? g_blob_lens[idx] : 0;
            memcpy(p, &dl, 4); p += 4;
            if (dl > 0) { memcpy(p, g_blobs[idx], dl); p += dl; }
        }
        if (shape & SHAPE_VECTOR) {
            if (read_vec_from_device(idx, (float *)p) < 0) { free(buf); return NULL; }
            p += g_dim * 4;
        }
    }
    *out_len = total;
    return buf;
}

static int process_binary_frame(unsigned char cmd, const char *label, int label_len,
                                const char *data, int data_len,
                                write_fn writer, void *wctx) {
    /* readonly check for write commands */
    if (g_readonly) {
        if (cmd == CMD_PUSH || cmd == CMD_UPDATE || cmd == CMD_DELETE ||
            cmd == CMD_LABEL || cmd == CMD_UNDO || cmd == CMD_SAVE || cmd == CMD_SET_DATA) {
            resp_err(writer, wctx, "read-only mode");
            return 0;
        }
    }

    switch (cmd) {

    case CMD_PUSH: {
        /* body: vec(dim*4) + 4B data_len + data_bytes; label via header. */
        int vbytes = g_dim * (int)sizeof(float);
        if (data_len < vbytes + 4) { resp_err(writer, wctx, "body too short"); return 0; }
        unsigned int dlen;
        memcpy(&dlen, data + vbytes, 4);
        if (data_len != vbytes + 4 + (int)dlen) { resp_err(writer, wctx, "body length mismatch"); return 0; }
        if (dlen > MAX_DATA_BYTES) { resp_err(writer, wctx, "data too large"); return 0; }
        if (dlen > 0 && label_len <= 0) { resp_err(writer, wctx, "data requires label"); return 0; }
        if (label_len > 0) {
            int rc = validate_label(label, label_len);
            if (rc == -1) { resp_err(writer, wctx, "label has invalid chars"); return 0; }
            if (rc == -2) { resp_err(writer, wctx, "label too long"); return 0; }
            if (rc == -3) { resp_err(writer, wctx, "label empty"); return 0; }
        }
        int slot = vec_push((const float *)data);
        if (label_len > 0) {
            if (vec_set_label(slot, label, label_len) != 0) {
                /* shouldn't happen — already validated. roll back. */
                g_count--;
                resp_err(writer, wctx, "label store failed");
                return 0;
            }
        }
        if (dlen > 0) {
            if (vec_set_blob(slot, (const unsigned char *)(data + vbytes + 4), dlen) != 0) {
                resp_err(writer, wctx, "data store failed");
                return 0;
            }
        }
        char body[4]; memcpy(body, &slot, 4);
        resp_ok_header(writer, wctx, 4);
        writer(wctx, body, 4);
        return 0;
    }

    case CMD_QUERY: {
        /* body: 1B metric + 1B shape + vec */
        int vbytes = g_dim * (int)sizeof(float);
        if (data_len != 1 + 1 + vbytes) { resp_err(writer, wctx, "bad query body"); return 0; }
        unsigned char metric = (unsigned char)data[0];
        unsigned char shape  = (unsigned char)data[1];
        if (metric > METRIC_COSINE) { resp_err(writer, wctx, "bad metric"); return 0; }
        const float *qv = (const float *)(data + 2);
        int ids[GPU_TOP_K]; float dists[GPU_TOP_K];
        int k = vec_pull(qv, ids, dists, metric);
        unsigned int blen;
        char *body = build_result_body(ids, dists, k, shape, 1, &blen);
        if (!body) { resp_err(writer, wctx, "out of memory"); return 0; }
        resp_ok_header(writer, wctx, blen);
        writer(wctx, body, (int)blen);
        free(body);
        return 0;
    }

    case CMD_QID: {
        /* body: 1B metric + 1B shape + (4B index | nothing if label_len>0) */
        if (data_len < 2) { resp_err(writer, wctx, "bad qid body"); return 0; }
        unsigned char metric = (unsigned char)data[0];
        unsigned char shape  = (unsigned char)data[1];
        if (metric > METRIC_COSINE) { resp_err(writer, wctx, "bad metric"); return 0; }
        int idx;
        if (label_len > 0) {
            if (data_len != 2) { resp_err(writer, wctx, "extra body bytes"); return 0; }
            idx = resolve_label(label, label_len);
            if (idx == -2) { resp_err(writer, wctx, "ambiguous label"); return 0; }
            if (idx < 0)   { resp_err(writer, wctx, "label not found"); return 0; }
        } else {
            if (data_len != 6) { resp_err(writer, wctx, "bad qid body"); return 0; }
            memcpy(&idx, data + 2, 4);
            if (idx < 0 || idx >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
            if (!g_alive[idx]) { resp_err(writer, wctx, "deleted"); return 0; }
        }
        int ids[GPU_TOP_K]; float dists[GPU_TOP_K];
        int k = vec_pull_by_idx(idx, ids, dists, metric);
        unsigned int blen;
        char *body = build_result_body(ids, dists, k, shape, 1, &blen);
        if (!body) { resp_err(writer, wctx, "out of memory"); return 0; }
        resp_ok_header(writer, wctx, blen);
        writer(wctx, body, (int)blen);
        free(body);
        return 0;
    }

    case CMD_GET: {
        /* body: 1B mode + 1B shape + (4B idx | 4B count + count*4B indices | nothing if label) */
        if (data_len < 2) { resp_err(writer, wctx, "bad get body"); return 0; }
        unsigned char mode  = (unsigned char)data[0];
        unsigned char shape = (unsigned char)data[1];
        int *match_ids = NULL;
        int match_count = 0;
        int allocated = 0;

        int single_id_buf;
        if (mode == GET_MODE_SINGLE) {
            if (label_len > 0) {
                if (data_len != 2) { resp_err(writer, wctx, "extra body bytes"); return 0; }
                match_count = resolve_label_all(label, label_len, &match_ids);
                if (match_count == 0) { resp_err(writer, wctx, "label not found"); return 0; }
                allocated = 1;
            } else {
                if (data_len != 6) { resp_err(writer, wctx, "bad get body"); return 0; }
                memcpy(&single_id_buf, data + 2, 4);
                if (single_id_buf < 0 || single_id_buf >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
                if (!g_alive[single_id_buf]) { resp_err(writer, wctx, "deleted"); return 0; }
                match_ids = &single_id_buf;
                match_count = 1;
            }
        } else if (mode == GET_MODE_BATCH) {
            if (data_len < 6) { resp_err(writer, wctx, "bad batch body"); return 0; }
            unsigned int n;
            memcpy(&n, data + 2, 4);
            if (data_len != (int)(2 + 4 + n * 4)) { resp_err(writer, wctx, "bad batch length"); return 0; }
            const int *src = (const int *)(data + 6);
            for (unsigned int i = 0; i < n; i++) {
                if (src[i] < 0 || src[i] >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
                if (!g_alive[src[i]]) { resp_err(writer, wctx, "deleted"); return 0; }
            }
            match_ids = (int *)src;
            match_count = (int)n;
        } else {
            resp_err(writer, wctx, "bad mode");
            return 0;
        }

        unsigned int blen;
        char *body = build_result_body(match_ids, NULL, match_count, shape, 0, &blen);
        if (allocated) free(match_ids);
        if (!body) { resp_err(writer, wctx, "out of memory"); return 0; }
        resp_ok_header(writer, wctx, blen);
        writer(wctx, body, (int)blen);
        free(body);
        return 0;
    }

    case CMD_UPDATE: {
        int expected = g_dim * (int)sizeof(float);
        int idx;
        const float *vec_data;
        if (label_len > 0) {
            if (data_len != expected) { resp_err(writer, wctx, "bad body length"); return 0; }
            idx = resolve_label(label, label_len);
            if (idx == -2) { resp_err(writer, wctx, "ambiguous label"); return 0; }
            if (idx < 0) { resp_err(writer, wctx, "label not found"); return 0; }
            vec_data = (const float *)data;
        } else {
            if (data_len != 4 + expected) { resp_err(writer, wctx, "bad body length"); return 0; }
            memcpy(&idx, data, 4);
            vec_data = (const float *)(data + 4);
        }
        if (idx < 0 || idx >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
        if (!g_alive[idx]) { resp_err(writer, wctx, "deleted"); return 0; }
        upload_and_store(vec_data, (char *)d_vectors + (size_t)idx * g_dim * g_elem_size, g_dim);
        g_dirty = 1; g_last_write = time(NULL);
        resp_ok_empty(writer, wctx);
        return 0;
    }

    case CMD_DELETE: {
        int idx;
        if (label_len > 0) {
            if (data_len != 0) { resp_err(writer, wctx, "extra body bytes"); return 0; }
            idx = resolve_label(label, label_len);
            if (idx == -2) { resp_err(writer, wctx, "ambiguous label"); return 0; }
            if (idx < 0) { resp_err(writer, wctx, "label not found"); return 0; }
        } else {
            if (data_len != 4) { resp_err(writer, wctx, "bad body length"); return 0; }
            memcpy(&idx, data, 4);
        }
        if (idx < 0 || idx >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
        if (!g_alive[idx]) { resp_err(writer, wctx, "already deleted"); return 0; }
        g_alive[idx] = 0;
        g_deleted++;
        g_dirty = 1; g_last_write = time(NULL);
        unsigned char zero = 0;
        CUDA_CHECK(cudaMemcpy(d_alive + idx, &zero, 1, cudaMemcpyHostToDevice));
        if (idx < g_labels_cap) { free(g_labels[idx]); g_labels[idx] = NULL; }
        if (idx < g_blobs_cap)  { free(g_blobs[idx]); g_blobs[idx] = NULL; g_blob_lens[idx] = 0; }
        resp_ok_empty(writer, wctx);
        return 0;
    }

    case CMD_LABEL: {
        if (data_len != 4) { resp_err(writer, wctx, "bad body length"); return 0; }
        int idx;
        memcpy(&idx, data, 4);
        if (idx < 0 || idx >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
        if (label_len > 0) {
            int rc = vec_set_label(idx, label, label_len);
            if (rc == -1) { resp_err(writer, wctx, "label has invalid chars"); return 0; }
            if (rc == -2) { resp_err(writer, wctx, "label too long"); return 0; }
            if (rc == -3) { resp_err(writer, wctx, "label empty"); return 0; }
        } else {
            vec_set_label(idx, NULL, 0);
        }
        g_dirty = 1; g_last_write = time(NULL);
        resp_ok_empty(writer, wctx);
        return 0;
    }

    case CMD_UNDO: {
        if (data_len != 0) { resp_err(writer, wctx, "extra body bytes"); return 0; }
        if (g_count == 0) { resp_err(writer, wctx, "empty"); return 0; }
        g_count--;
        if (!g_alive[g_count]) g_deleted--;
        g_alive[g_count] = 1;
        g_dirty = 1; g_last_write = time(NULL);
        unsigned char one = 1;
        CUDA_CHECK(cudaMemcpy(d_alive + g_count, &one, 1, cudaMemcpyHostToDevice));
        vec_set_label(g_count, NULL, 0);
        vec_set_blob(g_count, NULL, 0);
        resp_ok_empty(writer, wctx);
        return 0;
    }

    case CMD_SAVE: {
        if (data_len != 0) { resp_err(writer, wctx, "extra body bytes"); return 0; }
        save_to_file(1);
        char body[8];
        unsigned int saved = (unsigned int)g_count;
        unsigned int crc   = (unsigned int)g_loaded_crc;
        memcpy(body, &saved, 4);
        memcpy(body + 4, &crc, 4);
        resp_ok_header(writer, wctx, 8);
        writer(wctx, body, 8);
        return 0;
    }

    case CMD_INFO: {
        if (data_len != 0) { resp_err(writer, wctx, "extra body bytes"); return 0; }
        unsigned int name_len = (unsigned int)strlen(g_name);
        unsigned int blen = 4 + 4 + 4 + 1 + 8 + 4 + 1 + 4 + name_len + 1;
        char *body = (char *)malloc(blen);
        if (!body) { resp_err(writer, wctx, "out of memory"); return 0; }
        char *p = body;
        memcpy(p, &g_dim, 4); p += 4;
        memcpy(p, &g_count, 4); p += 4;
        memcpy(p, &g_deleted, 4); p += 4;
        *p++ = (char)g_fmt;
        long long mtime = (long long)g_last_write;
        memcpy(p, &mtime, 8); p += 8;
        memcpy(p, &g_loaded_crc, 4); p += 4;
        *p++ = (char)g_crc_ok;
        memcpy(p, &name_len, 4); p += 4;
        memcpy(p, g_name, name_len); p += name_len;
        *p++ = (char)PROTOCOL_VERSION;
        resp_ok_header(writer, wctx, blen);
        writer(wctx, body, (int)blen);
        free(body);
        return 0;
    }

    case CMD_CLUSTER: {
        if (data_len != 9) { resp_err(writer, wctx, "bad body length"); return 0; }
        float eps; memcpy(&eps, data, 4);
        unsigned char mode = (unsigned char)data[4];
        int min_pts; memcpy(&min_pts, data + 5, 4);
        if (eps <= 0.0f) { resp_err(writer, wctx, "invalid eps"); return 0; }
        if (mode > METRIC_COSINE) { resp_err(writer, wctx, "bad metric"); return 0; }
        if (min_pts < 1) min_pts = 1;
        /* legacy text writer — wrap into binary envelope using a buffer. */
        scratch_writer_ctx sctx; scratch_writer_init(&sctx);
        vec_cluster(eps, min_pts, mode, scratch_writer_fn, &sctx);
        resp_ok_header(writer, wctx, sctx.len);
        if (sctx.len > 0) writer(wctx, sctx.buf, (int)sctx.len);
        scratch_writer_free(&sctx);
        return 0;
    }

    case CMD_DISTINCT: {
        if (data_len != 5) { resp_err(writer, wctx, "bad body length"); return 0; }
        int k; memcpy(&k, data, 4);
        unsigned char mode = (unsigned char)data[4];
        if (k <= 0) { resp_err(writer, wctx, "invalid k"); return 0; }
        if (mode > METRIC_COSINE) { resp_err(writer, wctx, "bad metric"); return 0; }
        scratch_writer_ctx sctx; scratch_writer_init(&sctx);
        vec_distinct(k, mode, scratch_writer_fn, &sctx);
        resp_ok_header(writer, wctx, sctx.len);
        if (sctx.len > 0) writer(wctx, sctx.buf, (int)sctx.len);
        scratch_writer_free(&sctx);
        return 0;
    }

    case CMD_REPRESENT: {
        if (data_len != 9) { resp_err(writer, wctx, "bad body length"); return 0; }
        float eps; memcpy(&eps, data, 4);
        unsigned char mode = (unsigned char)data[4];
        int min_pts; memcpy(&min_pts, data + 5, 4);
        if (eps <= 0.0f) { resp_err(writer, wctx, "invalid eps"); return 0; }
        if (mode > METRIC_COSINE) { resp_err(writer, wctx, "bad metric"); return 0; }
        if (min_pts < 1) min_pts = 1;
        scratch_writer_ctx sctx; scratch_writer_init(&sctx);
        vec_represent(eps, min_pts, mode, scratch_writer_fn, &sctx);
        resp_ok_header(writer, wctx, sctx.len);
        if (sctx.len > 0) writer(wctx, sctx.buf, (int)sctx.len);
        scratch_writer_free(&sctx);
        return 0;
    }

    case CMD_SET_DATA: {
        int idx;
        unsigned int dlen;
        const unsigned char *bytes;
        if (label_len > 0) {
            if (data_len < 4) { resp_err(writer, wctx, "bad body length"); return 0; }
            memcpy(&dlen, data, 4);
            if (data_len != (int)(4 + dlen)) { resp_err(writer, wctx, "bad body length"); return 0; }
            idx = resolve_label(label, label_len);
            if (idx == -2) { resp_err(writer, wctx, "ambiguous label"); return 0; }
            if (idx < 0) { resp_err(writer, wctx, "label not found"); return 0; }
            bytes = (const unsigned char *)(data + 4);
        } else {
            if (data_len < 8) { resp_err(writer, wctx, "bad body length"); return 0; }
            memcpy(&idx, data, 4);
            memcpy(&dlen, data + 4, 4);
            if (data_len != (int)(8 + dlen)) { resp_err(writer, wctx, "bad body length"); return 0; }
            bytes = (const unsigned char *)(data + 8);
        }
        if (idx < 0 || idx >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
        if (!g_alive[idx]) { resp_err(writer, wctx, "deleted"); return 0; }
        if (dlen > MAX_DATA_BYTES) { resp_err(writer, wctx, "data too large"); return 0; }
        int rc = vec_set_blob(idx, dlen > 0 ? bytes : NULL, dlen);
        if (rc != 0) { resp_err(writer, wctx, "data store failed"); return 0; }
        g_dirty = 1; g_last_write = time(NULL);
        resp_ok_empty(writer, wctx);
        return 0;
    }

    case CMD_GET_DATA: {
        int idx;
        if (label_len > 0) {
            if (data_len != 0) { resp_err(writer, wctx, "extra body bytes"); return 0; }
            idx = resolve_label(label, label_len);
            if (idx == -2) { resp_err(writer, wctx, "ambiguous label"); return 0; }
            if (idx < 0) { resp_err(writer, wctx, "label not found"); return 0; }
        } else {
            if (data_len != 4) { resp_err(writer, wctx, "bad body length"); return 0; }
            memcpy(&idx, data, 4);
        }
        if (idx < 0 || idx >= g_count) { resp_err(writer, wctx, "index out of range"); return 0; }
        if (!g_alive[idx]) { resp_err(writer, wctx, "deleted"); return 0; }
        unsigned int dlen = (idx < g_blobs_cap && g_blobs[idx]) ? g_blob_lens[idx] : 0;
        resp_ok_header(writer, wctx, 4 + dlen);
        writer(wctx, (const char *)&dlen, 4);
        if (dlen > 0) writer(wctx, (const char *)g_blobs[idx], (int)dlen);
        return 0;
    }

    default:
        resp_err(writer, wctx, "unknown binary command");
        return 0;
    }
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
        char cnt[32], active[32], del[32], cap[32], rem[32], fsz[32];
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
    printf("start a database:\n");
    printf("  vec                                  find .tensors in current dir or create new\n");
    printf("  vec mydb 1024                        create/open 1024-dim database\n");
    printf("  vec mydb 1024:f16                    fp16 mode (half VRAM, double capacity)\n");
    printf("  vec mydb 1024 1921                   custom port\n");
    printf("  vec mydb.tensors                     load from file\n\n");
    printf("protocol (binary frame only, 0xF0 prefix):\n");
    printf("  all commands via structured binary frames over TCP, pipe, or socket\n");
    printf("  format: F0 <2B ns_len> [ns] <CMD> <2B label_len> [label] [data]\n\n");
    printf("multiple databases:\n");
    printf("  vec deploy                           auto-discover .tensors, launch all\n");
    printf("  vec --deploy=a:1024,b:512 1920       explicit schema, custom port\n");
    printf("  vec --notcp tools 1024               pipe/socket only (no TCP)\n");
    printf("  vec --route 1920                     route multiple --notcp instances\n\n");
    printf("housekeeping:\n");
    printf("  vec mydb --delete                    destroy database files\n");
    printf("  vec mydb --check                     verify file integrity (dry run)\n");
    printf("  vec mydb --repair                    verify and fix file integrity\n");
    printf("  vec --help                           this message\n\n");
}

/* ===================================================================== */
/*  File repair                                                          */
/* ===================================================================== */

static int run_repair(int dry_run) {
    FILE *f = fopen(g_filepath, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", g_filepath);
        return 1;
    }

    /* get file size */
    fseek(f, 0, SEEK_END);
    long long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* read header */
    int file_dim, file_count, file_deleted;
    unsigned char file_fmt;
    if (fread(&file_dim, sizeof(int), 1, f) != 1 ||
        fread(&file_count, sizeof(int), 1, f) != 1 ||
        fread(&file_deleted, sizeof(int), 1, f) != 1 ||
        fread(&file_fmt, 1, 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "ERROR: cannot read header - file corrupt\n");
        return 1;
    }

    int elem_size = (file_fmt == FMT_F16) ? 2 : 4;
    long long header_bytes = 13;
    long long mask_bytes = file_count;
    long long vec_bytes = (long long)file_count * file_dim * elem_size;
    long long expected_size = header_bytes + mask_bytes + vec_bytes + 4; /* +4 for CRC */

    printf("file:       %s\n", g_filepath);
    printf("size:       %lld bytes\n", file_size);
    printf("format:     %s, %d dim\n", file_fmt == FMT_F16 ? "f16" : "f32", file_dim);
    printf("header:     %d total, %d deleted, %d alive\n", file_count, file_deleted, file_count - file_deleted);
    printf("expected:   %lld bytes\n", expected_size);

    if (file_size < header_bytes + mask_bytes) {
        fclose(f);
        fprintf(stderr, "\nERROR: file too small - alive mask incomplete, cannot repair\n");
        return 1;
    }

    /* read alive mask */
    unsigned char *alive = (unsigned char *)malloc(file_count);
    if (!alive) {
        fclose(f);
        fprintf(stderr, "\nFATAL: out of CPU memory reading alive mask (%d bytes)\n", file_count);
        return 1;
    }
    size_t mask_rd = fread(alive, 1, file_count, f);
    if ((int)mask_rd != file_count) {
        fclose(f);
        free(alive);
        fprintf(stderr, "\nERROR: alive mask truncated (%d of %d bytes)\n", (int)mask_rd, file_count);
        return 1;
    }

    /* how many vectors can we actually read? */
    long long data_avail = file_size - header_bytes - mask_bytes;
    int stride = file_dim * elem_size;
    int full_vectors = (int)(data_avail / stride);
    int has_crc = 0;

    /* check if there's a CRC trailer */
    if (data_avail >= (long long)file_count * stride + 4) {
        full_vectors = file_count;
        has_crc = 1;
    } else if (data_avail >= (long long)file_count * stride) {
        full_vectors = file_count;
        /* might have partial CRC or no CRC */
        long long leftover = data_avail - (long long)file_count * stride;
        has_crc = (leftover >= 4) ? 1 : 0;
    } else {
        full_vectors = (int)(data_avail / stride);
    }

    int truncated = file_count - full_vectors;
    printf("vectors:    %d readable, %d truncated\n", full_vectors, truncated);

    /* read vector data and compute CRC */
    unsigned int crc = 0;
    crc = crc32_update(crc, alive, file_count);

    /* scan each vector for NaN/Inf */
    int nan_count = 0;
    int *nan_indices = NULL;
    size_t total_data = (size_t)full_vectors * stride;
    void *vec_data = malloc(total_data);
    size_t rd = fread(vec_data, 1, total_data, f);
    crc = crc32_update(crc, vec_data, rd);

    for (int i = 0; i < full_vectors; i++) {
        if (!alive[i]) continue; /* already dead, skip */
        int bad = 0;
        if (file_fmt == FMT_F32) {
            float *v = (float *)((char *)vec_data + (size_t)i * stride);
            for (int d = 0; d < file_dim; d++) {
                unsigned int bits; memcpy(&bits, &v[d], 4);
                unsigned int exp = (bits >> 23) & 0xFF;
                if (exp == 0xFF) { bad = 1; break; }
            }
        } else {
            /* f16: read as uint16, check for NaN/Inf (exponent = 0x1F) */
            unsigned short *v = (unsigned short *)((char *)vec_data + (size_t)i * stride);
            for (int d = 0; d < file_dim; d++) {
                unsigned short exp = (v[d] >> 10) & 0x1F;
                if (exp == 0x1F) { bad = 1; break; }
            }
        }
        if (bad) {
            int *tmp = (int *)realloc(nan_indices, (nan_count + 1) * sizeof(int));
            if (!tmp) { fprintf(stderr, "WARN: out of CPU memory tracking NaN indices, stopping scan\n"); break; }
            nan_indices = tmp;
            nan_indices[nan_count++] = i;
        }
    }

    /* CRC check */
    unsigned int stored_crc = 0;
    int crc_present = 0;
    if (has_crc && fread(&stored_crc, sizeof(unsigned int), 1, f) == 1) {
        crc_present = 1;
    }

    fclose(f);

    printf("CRC:        ");
    if (crc_present) {
        if (stored_crc == crc)
            printf("0x%08X - ok\n", stored_crc);
        else
            printf("MISMATCH - stored 0x%08X, computed 0x%08X\n", stored_crc, crc);
    } else {
        printf("missing (old format or truncated)\n");
    }

    printf("bad vectors: %d (NaN/Inf in alive entries)\n", nan_count);

    /* summary */
    int issues = truncated + nan_count + (crc_present && stored_crc != crc ? 1 : 0) + (!crc_present ? 1 : 0);

    if (issues == 0) {
        printf("\nfile is healthy, no repairs needed\n");
        free(alive);
        free(vec_data);
        free(nan_indices);
        return 0;
    }

    printf("\nissues found:\n");
    if (truncated > 0)
        printf("  - %d vectors truncated (indices %d..%d have no data)\n", truncated, full_vectors, file_count - 1);
    if (nan_count > 0) {
        printf("  - %d vectors contain NaN/Inf:", nan_count);
        for (int i = 0; i < nan_count && i < 20; i++) printf(" %d", nan_indices[i]);
        if (nan_count > 20) printf(" ... (%d more)", nan_count - 20);
        printf("\n");
    }
    if (!crc_present)
        printf("  - CRC trailer missing\n");
    else if (stored_crc != crc)
        printf("  - CRC mismatch\n");

    if (dry_run) {
        printf("\ndry run - no changes made. use --repair to fix.\n");
        free(alive);
        free(vec_data);
        free(nan_indices);
        return (issues > 0) ? 1 : 0;
    }

    /* repair: tombstone bad vectors, truncate count, rewrite with CRC */
    printf("\nrepairing...\n");

    int repaired = 0;

    /* tombstone NaN/Inf vectors */
    for (int i = 0; i < nan_count; i++) {
        int idx = nan_indices[i];
        if (alive[idx]) {
            alive[idx] = 0;
            file_deleted++;
            repaired++;
            printf("  tombstoned index %d (bad embedding)\n", idx);
        }
    }

    /* truncate count to what we can actually read */
    if (truncated > 0) {
        printf("  adjusted count %d -> %d (%d truncated entries removed)\n", file_count, full_vectors, truncated);
        file_count = full_vectors;
    }

    /* rewrite file */
    FILE *fw = fopen(g_filepath, "wb");
    if (!fw) {
        fprintf(stderr, "ERROR: cannot open %s for writing\n", g_filepath);
        free(alive); free(vec_data); free(nan_indices);
        return 1;
    }

    fwrite(&file_dim, sizeof(int), 1, fw);
    fwrite(&file_count, sizeof(int), 1, fw);
    fwrite(&file_deleted, sizeof(int), 1, fw);
    unsigned char fmt_byte = (unsigned char)file_fmt;
    fwrite(&fmt_byte, 1, 1, fw);

    unsigned int new_crc = 0;
    fwrite(alive, 1, file_count, fw);
    new_crc = crc32_update(new_crc, alive, file_count);

    size_t write_bytes = (size_t)file_count * stride;
    fwrite(vec_data, 1, write_bytes, fw);
    new_crc = crc32_update(new_crc, vec_data, write_bytes);

    fwrite(&new_crc, sizeof(unsigned int), 1, fw);
    fclose(fw);

    char crc_name[16];
    crc32_word(new_crc, crc_name);
    printf("\nrepaired: %d issue(s) fixed, %d vectors, checksum %s (0x%08X)\n",
           repaired + (truncated > 0 ? 1 : 0), file_count, crc_name, new_crc);

    free(alive);
    free(vec_data);
    free(nan_indices);
    return 0;
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
    int dim;
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

static int router_send_recv(route_entry *r, const char *data, int data_len,
                            const char *bin, int bin_len, char *resp, int resp_max) {
    if (!router_connect(r)) return -1;
    DWORD written;
    if (!WriteFile(r->pipe, data, data_len, &written, NULL)) {
        CloseHandle(r->pipe); r->pipe = INVALID_HANDLE_VALUE; r->connected = 0; return -1;
    }
    if (bin && bin_len > 0) {
        if (!WriteFile(r->pipe, bin, bin_len, &written, NULL)) {
            CloseHandle(r->pipe); r->pipe = INVALID_HANDLE_VALUE; r->connected = 0; return -1;
        }
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
            if (buf_used < 1) break;

            /* binary frame: extract namespace from frame, forward with ns_len=0 */
            if ((unsigned char)buf[0] == BIN_MAGIC) {
                if (buf_used < 3) break;
                unsigned short frame_ns_len; memcpy(&frame_ns_len, buf + 1, 2);
                int hdr = 3 + frame_ns_len;
                if (buf_used < hdr + 3) break;
                unsigned char cmd = (unsigned char)buf[hdr];
                unsigned short lbl_len; memcpy(&lbl_len, buf + hdr + 1, 2);
                int header_total = hdr + 3 + lbl_len + 4; /* +4 for explicit body_len (2.0) */
                if (buf_used < header_total) break;
                const char *data_start = buf + hdr + 3 + lbl_len;
                int available = buf_used - (hdr + 3 + lbl_len);
                int dlen = frame_data_len(cmd, data_start, available, lbl_len);
                if (dlen == -1) break;
                if (dlen == -2) {
                    /* unknown command — emit binary error envelope */
                    const char *err = "unknown binary command";
                    char ebuf[64]; ebuf[0] = RESP_ERR;
                    unsigned int el = (unsigned int)strlen(err);
                    memcpy(ebuf + 1, &el, 4);
                    memcpy(ebuf + 5, err, el);
                    send(client, ebuf, 5 + el, 0);
                    buf_used -= header_total;
                    if (buf_used > 0) memmove(buf, buf + header_total, buf_used);
                    continue;
                }
                int frame_total = header_total + dlen;
                while (buf_used < frame_total) {
                    r = recv(client, buf + buf_used, sizeof(buf) - buf_used, 0);
                    if (r <= 0) goto router_done;
                    buf_used += r;
                }
                /* find route from frame namespace */
                route_entry *route = NULL;
                if (frame_ns_len > 0) {
                    const char *ns = buf + 3;
                    for (int i = 0; i < g_route_count; i++) {
                        if ((int)strlen(g_routes[i].name) == frame_ns_len && strncmp(g_routes[i].name, ns, frame_ns_len) == 0) { route = &g_routes[i]; break; }
                    }
                    if (!route) {
                        int old_count = g_route_count;
                        router_discover_pipes();
                        for (int i = old_count; i < g_route_count; i++) {
                            if ((int)strlen(g_routes[i].name) == frame_ns_len && strncmp(g_routes[i].name, ns, frame_ns_len) == 0) { route = &g_routes[i]; break; }
                        }
                    }
                }
                char resp[65536];
                if (!route) {
                    const char *err = "unknown namespace";
                    char ebuf[64]; ebuf[0] = RESP_ERR;
                    unsigned int el = (unsigned int)strlen(err);
                    memcpy(ebuf + 1, &el, 4);
                    memcpy(ebuf + 5, err, el);
                    send(client, ebuf, 5 + el, 0);
                } else {
                    /* rebuild frame with ns_len=0, preserving explicit body_len */
                    int fwd_len = 1 + 2 + 1 + 2 + lbl_len + 4 + dlen;
                    char *fwd = (char *)malloc(fwd_len);
                    char *p = fwd;
                    *p++ = (char)BIN_MAGIC;
                    unsigned short zero_ns = 0; memcpy(p, &zero_ns, 2); p += 2;
                    *p++ = (char)cmd;
                    memcpy(p, &lbl_len, 2); p += 2;
                    if (lbl_len > 0) { memcpy(p, buf + hdr + 3, lbl_len); p += lbl_len; }
                    unsigned int blen_le = (unsigned int)dlen;
                    memcpy(p, &blen_le, 4); p += 4;
                    if (dlen > 0) { memcpy(p, buf + header_total, dlen); p += dlen; }
                    int rlen = router_send_recv(route, fwd, fwd_len, NULL, 0, resp, sizeof(resp));
                    free(fwd);
                    if (rlen > 0) send(client, resp, rlen, 0);
                    else {
                        char ebuf[256]; ebuf[0] = RESP_ERR;
                        int el = snprintf(ebuf + 5, sizeof(ebuf) - 5, "pipe disconnected '%s'", route->name);
                        unsigned int elu = (unsigned int)el;
                        memcpy(ebuf + 1, &elu, 4);
                        send(client, ebuf, 5 + el, 0);
                    }
                }
                buf_used -= frame_total;
                if (buf_used > 0) memmove(buf, buf + frame_total, buf_used);
                continue;
            }

            /* not a binary frame, skip to next 0xF0 */
            { int skip = 1;
              while (skip < buf_used && (unsigned char)buf[skip] != BIN_MAGIC) skip++;
              buf_used -= skip;
              if (buf_used > 0) memmove(buf, buf + skip, buf_used);
            }
        }
    }
router_done:
    closesocket(client);
    return 0;
}

static int run_router(int port, int from_deploy = 0) {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    router_discover_pipes();
    if (!from_deploy) {
        printf("vec router on port %d\n", port);
        printf("discovered %d namespace%s:\n", g_route_count, g_route_count == 1 ? "" : "s");
        for (int i = 0; i < g_route_count; i++) printf("  %s -> \\\\.\\pipe\\vec_%s\n", g_routes[i].name, g_routes[i].name);
        if (g_route_count == 0) printf("  (none found, will discover on first request)\n");
        printf("\n");
    }
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
    printf("TCP listening on 0.0.0.0:%d\n", port);
    printf("ready for connections, ctrl+c to exit\n");
    if (!from_deploy) printf("hint: use --deploy to automate instance spawning and routing\n");
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

static int router_send_recv(route_entry *r, const char *data, int data_len,
                            const char *bin, int bin_len, char *resp, int resp_max) {
    if (!router_connect(r)) return -1;
    if (send(r->fd, data, data_len, 0) <= 0) { close(r->fd); r->fd = -1; r->connected = 0; return -1; }
    if (bin && bin_len > 0) {
        if (send(r->fd, bin, bin_len, 0) <= 0) { close(r->fd); r->fd = -1; r->connected = 0; return -1; }
    }
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
            if (buf_used < 1) break;

            /* binary frame: extract namespace from frame, forward with ns_len=0 */
            if ((unsigned char)buf[0] == BIN_MAGIC) {
                if (buf_used < 3) break;
                unsigned short frame_ns_len; memcpy(&frame_ns_len, buf + 1, 2);
                int hdr = 3 + frame_ns_len;
                if (buf_used < hdr + 3) break;
                unsigned char cmd = (unsigned char)buf[hdr];
                unsigned short lbl_len; memcpy(&lbl_len, buf + hdr + 1, 2);
                int header_total = hdr + 3 + lbl_len + 4; /* +4 for explicit body_len (2.0) */
                if (buf_used < header_total) break;
                const char *data_start = buf + hdr + 3 + lbl_len;
                int available = buf_used - (hdr + 3 + lbl_len);
                int dlen = frame_data_len(cmd, data_start, available, lbl_len);
                if (dlen == -1) break;
                if (dlen == -2) {
                    const char *err = "unknown binary command";
                    char ebuf[64]; ebuf[0] = RESP_ERR;
                    unsigned int el = (unsigned int)strlen(err);
                    memcpy(ebuf + 1, &el, 4);
                    memcpy(ebuf + 5, err, el);
                    send(client, ebuf, 5 + el, 0);
                    buf_used -= header_total;
                    if (buf_used > 0) memmove(buf, buf + header_total, buf_used);
                    continue;
                }
                int frame_total = header_total + dlen;
                while (buf_used < frame_total) {
                    r = recv(client, buf + buf_used, sizeof(buf) - buf_used, 0);
                    if (r <= 0) goto router_done;
                    buf_used += r;
                }
                /* find route from frame namespace */
                route_entry *route = NULL;
                if (frame_ns_len > 0) {
                    const char *ns = buf + 3;
                    for (int i = 0; i < g_route_count; i++) {
                        if ((int)strlen(g_routes[i].name) == frame_ns_len && strncmp(g_routes[i].name, ns, frame_ns_len) == 0) { route = &g_routes[i]; break; }
                    }
                    if (!route) {
                        int old_count = g_route_count;
                        router_discover_sockets();
                        for (int i = old_count; i < g_route_count; i++) {
                            if ((int)strlen(g_routes[i].name) == frame_ns_len && strncmp(g_routes[i].name, ns, frame_ns_len) == 0) { route = &g_routes[i]; break; }
                        }
                    }
                }
                char resp[65536];
                if (!route) {
                    const char *err = "unknown namespace";
                    char ebuf[64]; ebuf[0] = RESP_ERR;
                    unsigned int el = (unsigned int)strlen(err);
                    memcpy(ebuf + 1, &el, 4);
                    memcpy(ebuf + 5, err, el);
                    send(client, ebuf, 5 + el, 0);
                } else {
                    /* rebuild frame with ns_len=0, preserving explicit body_len */
                    int fwd_len = 1 + 2 + 1 + 2 + lbl_len + 4 + dlen;
                    char *fwd = (char *)malloc(fwd_len);
                    char *p = fwd;
                    *p++ = (char)BIN_MAGIC;
                    unsigned short zero_ns = 0; memcpy(p, &zero_ns, 2); p += 2;
                    *p++ = (char)cmd;
                    memcpy(p, &lbl_len, 2); p += 2;
                    if (lbl_len > 0) { memcpy(p, buf + hdr + 3, lbl_len); p += lbl_len; }
                    unsigned int blen_le = (unsigned int)dlen;
                    memcpy(p, &blen_le, 4); p += 4;
                    if (dlen > 0) { memcpy(p, buf + header_total, dlen); p += dlen; }
                    int rlen = router_send_recv(route, fwd, fwd_len, NULL, 0, resp, sizeof(resp));
                    free(fwd);
                    if (rlen > 0) send(client, resp, rlen, 0);
                    else {
                        char ebuf[256]; ebuf[0] = RESP_ERR;
                        int el = snprintf(ebuf + 5, sizeof(ebuf) - 5, "socket disconnected '%s'", route->name);
                        unsigned int elu = (unsigned int)el;
                        memcpy(ebuf + 1, &elu, 4);
                        send(client, ebuf, 5 + el, 0);
                    }
                }
                buf_used -= frame_total;
                if (buf_used > 0) memmove(buf, buf + frame_total, buf_used);
                continue;
            }

            /* not a binary frame, skip to next 0xF0 */
            { int skip = 1;
              while (skip < buf_used && (unsigned char)buf[skip] != BIN_MAGIC) skip++;
              buf_used -= skip;
              if (buf_used > 0) memmove(buf, buf + skip, buf_used);
            }
        }
    }
router_done:
    close(client);
    return NULL;
}

static int run_router(int port, int from_deploy = 0) {
    router_discover_sockets();
    if (!from_deploy) {
        printf("vec router on port %d\n", port);
        printf("discovered %d namespace%s:\n", g_route_count, g_route_count == 1 ? "" : "s");
        for (int i = 0; i < g_route_count; i++) printf("  %s -> /tmp/vec_%s.sock\n", g_routes[i].name, g_routes[i].name);
        if (g_route_count == 0) printf("  (none found, will discover on first request)\n");
        printf("\n");
    }
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
    printf("TCP listening on 0.0.0.0:%d\n", port);
    printf("ready for connections, ctrl+c to exit\n");
    if (!from_deploy) printf("hint: use --deploy to automate instance spawning and routing\n");
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

/* ===================================================================== */
/*  Deploy mode (shared)                                                 */
/* ===================================================================== */

static void generate_random_name(char *buf, int len); /* defined per-platform */

#define MAX_DEPLOY 32

struct deploy_entry {
    char name[64];
    int  dim;
    char format[8];
};

static deploy_entry g_deploy[MAX_DEPLOY];
static int g_deploy_count = 0;

static int parse_deploy(const char *spec) {
    char buf[2048];
    strncpy(buf, spec, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *saveptr = NULL;
    char *tok = strtok_r(buf, ",", &saveptr);
    while (tok && g_deploy_count < MAX_DEPLOY) {
        deploy_entry *e = &g_deploy[g_deploy_count];
        memset(e, 0, sizeof(*e));

        char *c1 = strchr(tok, ':');
        if (!c1) {
            fprintf(stderr, "ERROR: deploy entry '%s' missing dimension (use name:dim)\n", tok);
            return -1;
        }
        int nlen = (int)(c1 - tok);
        if (nlen <= 0 || nlen >= (int)sizeof(e->name)) {
            fprintf(stderr, "ERROR: invalid name in deploy entry '%s'\n", tok);
            return -1;
        }
        memcpy(e->name, tok, nlen);
        e->name[nlen] = '\0';

        char *after_dim = c1 + 1;
        char *c2 = strchr(after_dim, ':');
        if (c2) {
            *c2 = '\0';
            strncpy(e->format, c2 + 1, sizeof(e->format) - 1);
        }
        e->dim = atoi(after_dim);
        if (e->dim <= 0) {
            fprintf(stderr, "ERROR: invalid dimension in deploy entry '%s'\n", tok);
            return -1;
        }
        for (int j = 0; j < g_deploy_count; j++) {
            if (strcmp(g_deploy[j].name, e->name) == 0) {
                fprintf(stderr, "ERROR: duplicate database name '%s'\n", e->name);
                return -1;
            }
        }
        g_deploy_count++;
        tok = strtok_r(NULL, ",", &saveptr);
    }
    return g_deploy_count;
}

static void write_empty_tensors(deploy_entry *e) {
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s_%d_%s.tensors",
             e->name, e->dim, (e->format[0] ? e->format : "f32"));
    FILE *f = fopen(filepath, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot create %s\n", filepath); return; }
    int zero = 0;
    unsigned char fmt = (strcmp(e->format, "f16") == 0) ? FMT_F16 : FMT_F32;
    fwrite(&e->dim, sizeof(int), 1, f);
    fwrite(&zero, sizeof(int), 1, f);     /* count */
    fwrite(&zero, sizeof(int), 1, f);     /* deleted */
    fwrite(&fmt, 1, 1, f);
    unsigned int crc = 0;
    fwrite(&crc, sizeof(unsigned int), 1, f);
    fclose(f);
}

static int deploy_wizard(int has_f16) {
    printf("=== vec deploy wizard ===\n\n");

    while (g_deploy_count < MAX_DEPLOY) {
        deploy_entry *e = &g_deploy[g_deploy_count];
        memset(e, 0, sizeof(*e));
        char line[256];

        printf("database #%d\n", g_deploy_count + 1);

    name_retry:
        printf("  name [random]: ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0]) {
            strncpy(e->name, line, sizeof(e->name) - 1);
        } else {
            generate_random_name(e->name, 6);
            printf("  -> %s\n", e->name);
        }
        for (int j = 0; j < g_deploy_count; j++) {
            if (strcmp(g_deploy[j].name, e->name) == 0) {
                printf("  '%s' already exists, pick another\n", e->name);
                goto name_retry;
            }
        }

        printf("  dim [1024]: ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        line[strcspn(line, "\r\n")] = '\0';
        e->dim = line[0] ? atoi(line) : 1024;
        if (e->dim <= 0) e->dim = 1024;

        if (has_f16) {
            printf("  format [f32]: ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\r\n")] = '\0';
            if (line[0] == '1' || strcmp(line, "f16") == 0 || strcmp(line, "fp16") == 0)
                strncpy(e->format, "f16", sizeof(e->format) - 1);
        }

        g_deploy_count++;
        printf("\n");

        if (g_deploy_count >= 2) {
            printf("add another? (y/n) [n]: ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\r\n")] = '\0';
            if (line[0] != 'y' && line[0] != 'Y') break;
            printf("\n");
        }
    }

    if (g_deploy_count == 0) return 0;

    printf("\ncreating %d database%s...\n", g_deploy_count, g_deploy_count == 1 ? "" : "s");
    for (int i = 0; i < g_deploy_count; i++) {
        write_empty_tensors(&g_deploy[i]);
        printf("  %s_%d_%s.tensors\n", g_deploy[i].name, g_deploy[i].dim,
               g_deploy[i].format[0] ? g_deploy[i].format : "f32");
    }
    printf("\n");

    return g_deploy_count;
}

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

struct db_entry { char name[256]; int dim; int fmt; char filename[512]; long long file_size; };

static int find_any_db() {
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA("*.tensors", &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;

    db_entry entries[64];
    int count = 0;

    do {
        if (count >= 64) break;
        char buf[256];
        strncpy(buf, fd.cFileName, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
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

        strncpy(entries[count].name, buf, sizeof(entries[count].name) - 1);
        entries[count].dim = d;
        entries[count].fmt = (strcmp(fs, "f16") == 0) ? FMT_F16 : FMT_F32;
        strncpy(entries[count].filename, fd.cFileName, sizeof(entries[count].filename) - 1);
        LARGE_INTEGER sz;
        sz.LowPart = fd.nFileSizeLow;
        sz.HighPart = fd.nFileSizeHigh;
        entries[count].file_size = sz.QuadPart;
        count++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);

    if (count == 0) return 0;

    int pick = 0;
    if (count > 1) {
        printf("databases found:\n");
        for (int i = 0; i < count; i++) {
            char sz[32];
            fmt_bytes(sz, sizeof(sz), (double)entries[i].file_size);
            printf("  [%d] %s (%d dim, %s, %s)\n", i + 1, entries[i].name,
                   entries[i].dim, entries[i].fmt == FMT_F16 ? "f16" : "f32", sz);
        }
        printf("select [1-%d]: ", count);
        fflush(stdout);
        char input[16];
        if (!fgets(input, sizeof(input), stdin)) return 0;
        pick = atoi(input) - 1;
        if (pick < 0 || pick >= count) {
            fprintf(stderr, "invalid selection\n");
            return 0;
        }
    }

    strncpy(g_name, entries[pick].name, sizeof(g_name) - 1);
    g_dim = entries[pick].dim;
    if (entries[pick].fmt == FMT_F16) { g_fmt = FMT_F16; g_elem_size = 2; }
    else { g_fmt = FMT_F32; g_elem_size = 4; }
    build_filepath();
    return 1;
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

static int discover_all_dbs() {
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA("*.tensors", &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;

    do {
        if (g_deploy_count >= MAX_DEPLOY) break;
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

        deploy_entry *e = &g_deploy[g_deploy_count];
        memset(e, 0, sizeof(*e));
        strncpy(e->name, buf, sizeof(e->name) - 1);
        e->dim = d;
        if (strcmp(fs, "f16") == 0) strncpy(e->format, "f16", sizeof(e->format) - 1);
        g_deploy_count++;
    } while (FindNextFileA(h, &fd));

    FindClose(h);
    return g_deploy_count;
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
            if (buf_used < 1) break;

            /* binary frame? */
            if ((unsigned char)buf[0] == BIN_MAGIC) {
                if (buf_used < 3) break;
                unsigned short ns_len; memcpy(&ns_len, buf + 1, 2);
                int hdr = 3 + ns_len;
                if (buf_used < hdr + 3) break;
                unsigned char cmd = (unsigned char)buf[hdr];
                unsigned short lbl_len; memcpy(&lbl_len, buf + hdr + 1, 2);
                int header_total = hdr + 3 + lbl_len + 4; /* +4 for explicit body_len (2.0) */
                if (buf_used < header_total) break;
                const char *label = (lbl_len > 0) ? buf + hdr + 3 : NULL;
                const char *data_start = buf + hdr + 3 + lbl_len;
                int available = buf_used - (hdr + 3 + lbl_len);
                int dlen = frame_data_len(cmd, data_start, available, lbl_len);
                if (dlen == -1) break;
                if (dlen == -2) {
                    const char *err = "unknown binary command";
                    char ebuf[64]; ebuf[0] = RESP_ERR;
                    unsigned int el = (unsigned int)strlen(err);
                    memcpy(ebuf + 1, &el, 4);
                    memcpy(ebuf + 5, err, el);
                    send(client, ebuf, 5 + el, 0);
                    buf_used -= header_total;
                    if (buf_used > 0) memmove(buf, buf + header_total, buf_used);
                    continue;
                }
                int frame_total = header_total + dlen;
                while (buf_used < frame_total && g_running) {
                    r = recv(client, buf + buf_used, MAX_LINE - buf_used, 0);
                    if (r <= 0) goto done;
                    buf_used += r;
                }
                REQ_LOCK();
                process_binary_frame(cmd, label, lbl_len, buf + header_total, dlen, tcp_writer, &client);
                REQ_UNLOCK();
                buf_used -= frame_total;
                if (buf_used > 0) memmove(buf, buf + frame_total, buf_used);
                continue;
            }

            /* not a binary frame, skip to next 0xF0 */
            { int skip = 1;
              while (skip < buf_used && (unsigned char)buf[skip] != BIN_MAGIC) skip++;
              buf_used -= skip;
              if (buf_used > 0) memmove(buf, buf + skip, buf_used);
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
                if (buf_used < 1) break;

                /* binary frame? */
                if ((unsigned char)buf[0] == BIN_MAGIC) {
                    if (buf_used < 3) break;
                    unsigned short ns_len; memcpy(&ns_len, buf + 1, 2);
                    int hdr = 3 + ns_len;
                    if (buf_used < hdr + 3) break;
                    unsigned char cmd = (unsigned char)buf[hdr];
                    unsigned short lbl_len; memcpy(&lbl_len, buf + hdr + 1, 2);
                    int header_total = hdr + 3 + lbl_len + 4; /* +4 for explicit body_len (2.0) */
                    if (buf_used < header_total) break;
                    const char *label = (lbl_len > 0) ? buf + hdr + 3 : NULL;
                    const char *data_start = buf + hdr + 3 + lbl_len;
                    int available = buf_used - (hdr + 3 + lbl_len);
                    int dlen = frame_data_len(cmd, data_start, available, lbl_len);
                    if (dlen == -1) break;
                    if (dlen == -2) {
                        const char *err = "unknown binary command";
                        char ebuf[64]; ebuf[0] = RESP_ERR;
                        unsigned int el = (unsigned int)strlen(err);
                        memcpy(ebuf + 1, &el, 4);
                        memcpy(ebuf + 5, err, el);
                        pipe_writer(&pipe, ebuf, 5 + (int)el);
                        buf_used -= header_total;
                        if (buf_used > 0) memmove(buf, buf + header_total, buf_used);
                        continue;
                    }
                    int frame_total = header_total + dlen;
                    while (buf_used < frame_total && g_running) {
                        ok = ReadFile(pipe, buf + buf_used, MAX_LINE - buf_used, &bytesRead, NULL);
                        if (!ok || bytesRead == 0) goto pipe_done;
                        buf_used += (int)bytesRead;
                    }
                    REQ_LOCK();
                    process_binary_frame(cmd, label, lbl_len, buf + header_total, dlen, pipe_writer, &pipe);
                    REQ_UNLOCK();
                    buf_used -= frame_total;
                    if (buf_used > 0) memmove(buf, buf + frame_total, buf_used);
                    continue;
                }

                /* not a binary frame, discard byte */
                buf_used--;
                if (buf_used > 0) memmove(buf, buf + 1, buf_used);
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

/* ----- Deploy mode (Windows) ----------------------------------------- */

struct deploy_child {
    HANDLE hProcess;
    HANDLE hStdoutRead;
    HANDLE hStderrRead;
    char   name[64];
    volatile int ready;
};

static deploy_child g_children[MAX_DEPLOY];
static int g_child_count = 0;

static DWORD WINAPI deploy_stdout_reader(LPVOID param) {
    deploy_child *child = (deploy_child *)param;
    char buf[4096];
    DWORD bytesRead;
    while (ReadFile(child->hStdoutRead, buf, sizeof(buf) - 1, &bytesRead, NULL) && bytesRead > 0) {
        for (DWORD i = 0; i < bytesRead; i++) {
            if (buf[i] == '\x06') child->ready = 1;
        }
    }
    return 0;
}

static DWORD WINAPI deploy_stderr_reader(LPVOID param) {
    deploy_child *child = (deploy_child *)param;
    char buf[4096];
    DWORD bytesRead;
    while (ReadFile(child->hStderrRead, buf, sizeof(buf) - 1, &bytesRead, NULL) && bytesRead > 0) {
        buf[bytesRead] = '\0';
        fprintf(stderr, "[%s] %s", child->name, buf);
    }
    return 0;
}

static void deploy_terminate_children() {
    for (int i = 0; i < g_child_count; i++) {
        if (g_children[i].hProcess) {
            GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, GetProcessId(g_children[i].hProcess));
        }
    }
    for (int i = 0; i < g_child_count; i++) {
        if (!g_children[i].hProcess) continue;
        printf("  %s... ", g_children[i].name);
        fflush(stdout);
        DWORD wait = WaitForSingleObject(g_children[i].hProcess, 60000);
        if (wait == WAIT_OBJECT_0) {
            printf("ok\n");
        } else {
            TerminateProcess(g_children[i].hProcess, 1);
            printf("killed\n");
        }
        CloseHandle(g_children[i].hProcess);
    }
}

static BOOL WINAPI deploy_ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        printf("\nshutting down deploy...\n");
        deploy_terminate_children();
        printf("deploy shutdown complete\n");
        ExitProcess(0);
        return TRUE;
    }
    return FALSE;
}

static int deploy_spawn_child(const char *exepath, deploy_entry *e) {
    deploy_child *c = &g_children[g_child_count];
    memset(c, 0, sizeof(*c));
    strncpy(c->name, e->name, sizeof(c->name) - 1);

    char cmdline[1024];
    if (e->format[0])
        snprintf(cmdline, sizeof(cmdline), "\"%s\" --notcp --nomotd %s %d:%s", exepath, e->name, e->dim, e->format);
    else
        snprintf(cmdline, sizeof(cmdline), "\"%s\" --notcp --nomotd %s %d", exepath, e->name, e->dim);

    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hStdoutRead, hStdoutWrite, hStderrRead, hStderrWrite;
    CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0);
    SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);
    CreatePipe(&hStderrRead, &hStderrWrite, &sa, 0);
    SetHandleInformation(hStderrRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = hStderrWrite;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    memset(&pi, 0, sizeof(pi));

    if (!CreateProcessA(NULL, cmdline, NULL, NULL, TRUE,
                        CREATE_NEW_PROCESS_GROUP, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "\n  ERROR: failed to spawn '%s': %lu\n", e->name, GetLastError());
        CloseHandle(hStdoutRead); CloseHandle(hStdoutWrite);
        CloseHandle(hStderrRead); CloseHandle(hStderrWrite);
        return 0;
    }

    CloseHandle(hStdoutWrite);
    CloseHandle(hStderrWrite);
    CloseHandle(pi.hThread);

    c->hProcess = pi.hProcess;
    c->hStdoutRead = hStdoutRead;
    c->hStderrRead = hStderrRead;
    c->ready = 0;
    g_child_count++;

    CreateThread(NULL, 0, deploy_stdout_reader, c, 0, NULL);
    CreateThread(NULL, 0, deploy_stderr_reader, c, 0, NULL);

    /* wait for ready */
    while (!c->ready) {
        DWORD exitCode;
        if (GetExitCodeProcess(c->hProcess, &exitCode) && exitCode != STILL_ACTIVE) {
            fprintf(stderr, "\n  ERROR: exited with code %lu\n", exitCode);
            return 0;
        }
        Sleep(50);
    }
    return 1;
}

static int run_deploy(int port) {
    char exepath[512];
    GetModuleFileNameA(NULL, exepath, sizeof(exepath));

    SetConsoleCtrlHandler(deploy_ctrl_handler, TRUE);

    /* find longest name for alignment */
    int maxlen = 0;
    for (int i = 0; i < g_deploy_count; i++) {
        int len = (int)strlen(g_deploy[i].name);
        if (len > maxlen) maxlen = len;
    }

    printf("loading databases...\n");
    for (int i = 0; i < g_deploy_count; i++) {
        deploy_entry *e = &g_deploy[i];
        printf("  %s%*s", e->name, maxlen - (int)strlen(e->name) + 3, "");
        fflush(stdout);
        if (!deploy_spawn_child(exepath, e)) {
            deploy_terminate_children();
            return 1;
        }
        printf("ok\n");
    }

    /* print pipe endpoints */
    for (int i = 0; i < g_deploy_count; i++)
        printf("Pipe listening on \\\\.\\pipe\\vec_%s\n", g_deploy[i].name);

    return run_router(port, 1);
}

/* ----- Signal + Main (Windows) --------------------------------------- */

static BOOL WINAPI ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        if (!g_running) return TRUE;
        g_running = 0;
        printf("\nshutting down...\n");
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
        if (strncmp(argv[i], "--deploy=", 9) == 0) {
            if (parse_deploy(argv[i] + 9) <= 0) return 1;
            int dport = DEFAULT_PORT;
            if (i + 1 < argc) { int p = atoi(argv[i + 1]); if (p > 0 && p <= 65535) dport = p; }
            return run_deploy(dport);
        }

        if (strcmp(argv[i], "deploy") == 0) {
            int add_mode = (i + 1 < argc && strcmp(argv[i + 1], "--add") == 0);
            int found = discover_all_dbs();
            if (add_mode) {
                /* discover existing, then wizard for one new db */
                deploy_wizard(1);
                if (g_deploy_count <= found) { fprintf(stderr, "no database added\n"); return 1; }
                /* create .tensors files for new entries */
                for (int j = found; j < g_deploy_count; j++)
                    write_empty_tensors(&g_deploy[j]);
            } else if (found <= 0) {
                found = deploy_wizard(1);
                if (found <= 0) return 1;
                system("cls");
                found = g_deploy_count;
            }
            if (g_deploy_count == 1) {
                fprintf(stderr, "WARNING: deploy is for multiple databases, launching single instance\n");
                argc = 1; /* fall through to normal single-db startup */
                break;
            }
            int dport = DEFAULT_PORT;
            for (int j = i + 1; j < argc; j++) {
                if (strcmp(argv[j], "--add") == 0) continue;
                int p = atoi(argv[j]);
                if (p > 0 && p <= 65535) { dport = p; break; }
            }
            return run_deploy(dport);
        }

        if (strcmp(argv[i], "--delete") == 0) {
            if (i < 2 || strcmp(argv[1], "--delete") == 0) {
                fprintf(stderr, "usage: vec <name> --delete\n");
                return 1;
            }
            const char *name = argv[1];
            WIN32_FIND_DATAA fd;
            char pattern[512];
            int deleted = 0;

            snprintf(pattern, sizeof(pattern), "%s_*.tensors", name);
            HANDLE h = FindFirstFileA(pattern, &fd);
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    if (DeleteFileA(fd.cFileName)) {
                        printf("deleted %s\n", fd.cFileName);
                        deleted++;
                    } else {
                        fprintf(stderr, "ERROR: could not delete %s\n", fd.cFileName);
                    }
                } while (FindNextFileA(h, &fd));
                FindClose(h);
            }

            snprintf(pattern, sizeof(pattern), "%s_*.meta", name);
            h = FindFirstFileA(pattern, &fd);
            if (h != INVALID_HANDLE_VALUE) {
                do {
                    if (DeleteFileA(fd.cFileName)) {
                        printf("deleted %s\n", fd.cFileName);
                        deleted++;
                    } else {
                        fprintf(stderr, "ERROR: could not delete %s\n", fd.cFileName);
                    }
                } while (FindNextFileA(h, &fd));
                FindClose(h);
            }

            if (deleted == 0) {
                fprintf(stderr, "no database files found for '%s'\n", name);
                return 1;
            }
            printf("destroyed %d file(s) for database '%s'\n", deleted, name);
            return 0;
        }

        if (strcmp(argv[i], "--check") == 0 || strcmp(argv[i], "--repair") == 0) {
            int dry_run = (strcmp(argv[i], "--check") == 0);
            if (i >= 2) {
                strncpy(g_name, argv[1], sizeof(g_name) - 1);
                if (!find_existing_db(argv[1])) {
                    fprintf(stderr, "ERROR: no database found for '%s'\n", argv[1]);
                    return 1;
                }
                return run_repair(dry_run);
            }
            WIN32_FIND_DATAA cfd;
            HANDLE ch = FindFirstFileA("*.tensors", &cfd);
            if (ch == INVALID_HANDLE_VALUE) {
                fprintf(stderr, "no .tensors files found in current directory\n");
                return 1;
            }
            int total = 0, errors = 0;
            do {
                strncpy(g_filepath, cfd.cFileName, sizeof(g_filepath) - 1);
                printf("--- %s ---\n", g_filepath);
                int r = run_repair(dry_run);
                if (r != 0) errors++;
                total++;
                printf("\n");
            } while (FindNextFileA(ch, &cfd));
            FindClose(ch);
            printf("checked %d database(s), %d with issues\n", total, errors);
            return (errors > 0) ? 1 : 0;
        }
    }

    int no_tcp = 0;
    int no_motd = 0;
    if (argc > 1 && strcmp(argv[1], "--notcp") == 0) { no_tcp = 1; argc--; argv++; }
    if (argc > 1 && strcmp(argv[1], "--nomotd") == 0) { no_motd = 1; argc--; argv++; }

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

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    double vram_gb = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    double max_records = (prop.totalGlobalMem * 0.9) / ((double)g_dim * g_elem_size);

    if (!no_motd) printf("%s (%.1f GB)\n", prop.name, vram_gb);

    gpu_init();
    cudaGetLastError();
    InitializeCriticalSection(&g_req_mutex);

    launch_l2_f32((const float *)d_vectors, (const float *)d_query, d_dists, 1, g_dim);
    cudaDeviceSynchronize();
    cudaGetLastError();

    int lr = load_from_file();
    if (lr < 0) {
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }

    /* check write permission without creating the file */
    {
        FILE *tf = fopen(g_filepath, "r+b"); /* existing file: open for update */
        if (tf) { fclose(tf); }
        else if (errno == ENOENT) {
            /* file doesn't exist yet — probe directory writability via a temp file */
            char probe[520];
            snprintf(probe, sizeof(probe), "_wprb%u.tmp", (unsigned)time(NULL));
            FILE *pf = fopen(probe, "wb");
            if (pf) { fclose(pf); remove(probe); }
            else { g_readonly = 1; }
        } else { g_readonly = 1; }
    }

    if (!no_motd) print_startup_info(file_exists, lr, max_records);
    if (g_readonly) printf("WARNING: read-only mode (cannot write to %s)\n", g_filepath);

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
    if (no_motd) {
        putchar('\x06');
        fflush(stdout);
    } else {
        printf("ready for connections, ctrl+c to exit\n");
        if (no_tcp)
            printf("hint: use --route to expose this instance via TCP\n");
    }

    while (g_running) {
        Sleep(500);
        if (g_dirty && g_last_write && (time(NULL) - g_last_write >= AUTOSAVE_IDLE_SECS)) {
            REQ_LOCK();
            if (g_dirty && (time(NULL) - g_last_write >= AUTOSAVE_IDLE_SECS))
                save_to_file(1);
            REQ_UNLOCK();
        }
    }

    if (h_tcp) WaitForSingleObject(h_tcp, 3000);
    WaitForSingleObject(h_pipe, 3000);

    save_to_file(0);
    DeleteCriticalSection(&g_req_mutex);
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

    db_entry entries[64];
    int count = 0;

    for (size_t i = 0; i < gl.gl_pathc && count < 64; i++) {
        const char *path = gl.gl_pathv[i];
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;

        char buf[256];
        strncpy(buf, base, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
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

        strncpy(entries[count].name, buf, sizeof(entries[count].name) - 1);
        entries[count].dim = d;
        entries[count].fmt = (strcmp(fs, "f16") == 0) ? FMT_F16 : FMT_F32;
        strncpy(entries[count].filename, path, sizeof(entries[count].filename) - 1);
        struct stat st;
        entries[count].file_size = (stat(path, &st) == 0) ? st.st_size : 0;
        count++;
    }
    globfree(&gl);

    if (count == 0) return 0;

    int pick = 0;
    if (count > 1) {
        printf("databases found:\n");
        for (int i = 0; i < count; i++) {
            char sz[32];
            fmt_bytes(sz, sizeof(sz), (double)entries[i].file_size);
            printf("  [%d] %s (%d dim, %s, %s)\n", i + 1, entries[i].name,
                   entries[i].dim, entries[i].fmt == FMT_F16 ? "f16" : "f32", sz);
        }
        printf("select [1-%d]: ", count);
        fflush(stdout);
        char input[16];
        if (!fgets(input, sizeof(input), stdin)) return 0;
        pick = atoi(input) - 1;
        if (pick < 0 || pick >= count) {
            fprintf(stderr, "invalid selection\n");
            return 0;
        }
    }

    strncpy(g_name, entries[pick].name, sizeof(g_name) - 1);
    g_dim = entries[pick].dim;
    if (entries[pick].fmt == FMT_F16) { g_fmt = FMT_F16; g_elem_size = 2; }
    else { g_fmt = FMT_F32; g_elem_size = 4; }
    build_filepath();
    return 1;
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

static int discover_all_dbs() {
    glob_t gl;
    if (glob("*.tensors", 0, NULL, &gl) != 0) return 0;

    for (size_t i = 0; i < gl.gl_pathc && g_deploy_count < MAX_DEPLOY; i++) {
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

        deploy_entry *e = &g_deploy[g_deploy_count];
        memset(e, 0, sizeof(*e));
        strncpy(e->name, buf, sizeof(e->name) - 1);
        e->dim = d;
        if (strcmp(fs, "f16") == 0) strncpy(e->format, "f16", sizeof(e->format) - 1);
        g_deploy_count++;
    }

    globfree(&gl);
    return g_deploy_count;
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
            if (buf_used < 1) break;

            /* binary frame? */
            if ((unsigned char)buf[0] == BIN_MAGIC) {
                if (buf_used < 3) break;
                unsigned short ns_len; memcpy(&ns_len, buf + 1, 2);
                int hdr = 3 + ns_len;
                if (buf_used < hdr + 3) break;
                unsigned char cmd = (unsigned char)buf[hdr];
                unsigned short lbl_len; memcpy(&lbl_len, buf + hdr + 1, 2);
                int header_total = hdr + 3 + lbl_len + 4; /* +4 for explicit body_len (2.0) */
                if (buf_used < header_total) break;
                const char *label = (lbl_len > 0) ? buf + hdr + 3 : NULL;
                const char *data_start = buf + hdr + 3 + lbl_len;
                int available = buf_used - (hdr + 3 + lbl_len);
                int dlen = frame_data_len(cmd, data_start, available, lbl_len);
                if (dlen == -1) break;
                if (dlen == -2) {
                    const char *err = "unknown binary command";
                    char ebuf[64]; ebuf[0] = RESP_ERR;
                    unsigned int el = (unsigned int)strlen(err);
                    memcpy(ebuf + 1, &el, 4);
                    memcpy(ebuf + 5, err, el);
                    send(client, ebuf, 5 + el, 0);
                    buf_used -= header_total;
                    if (buf_used > 0) memmove(buf, buf + header_total, buf_used);
                    continue;
                }
                int frame_total = header_total + dlen;
                while (buf_used < frame_total && g_running) {
                    r = recv(client, buf + buf_used, MAX_LINE - buf_used, 0);
                    if (r <= 0) goto done;
                    buf_used += r;
                }
                REQ_LOCK();
                process_binary_frame(cmd, label, lbl_len, buf + header_total, dlen, sock_writer, &client);
                REQ_UNLOCK();
                buf_used -= frame_total;
                if (buf_used > 0) memmove(buf, buf + frame_total, buf_used);
                continue;
            }

            /* not a binary frame, skip to next 0xF0 */
            { int skip = 1;
              while (skip < buf_used && (unsigned char)buf[skip] != BIN_MAGIC) skip++;
              buf_used -= skip;
              if (buf_used > 0) memmove(buf, buf + skip, buf_used);
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

/* ----- Deploy mode (Linux) ------------------------------------------- */

struct deploy_child {
    pid_t  pid;
    int    stdout_fd;
    int    stderr_fd;
    char   name[64];
    volatile int ready;
};

static deploy_child g_children[MAX_DEPLOY];
static int g_child_count = 0;

static void *deploy_stdout_reader(void *param) {
    deploy_child *child = (deploy_child *)param;
    char buf[4096];
    while (1) {
        ssize_t rd = read(child->stdout_fd, buf, sizeof(buf) - 1);
        if (rd <= 0) break;
        for (ssize_t i = 0; i < rd; i++) {
            if (buf[i] == '\x06') child->ready = 1;
        }
    }
    return NULL;
}

static void *deploy_stderr_reader(void *param) {
    deploy_child *child = (deploy_child *)param;
    char buf[4096];
    while (1) {
        ssize_t rd = read(child->stderr_fd, buf, sizeof(buf) - 1);
        if (rd <= 0) break;
        buf[rd] = '\0';
        fprintf(stderr, "[%s] %s", child->name, buf);
    }
    return NULL;
}

static void deploy_terminate_children() {
    for (int i = 0; i < g_child_count; i++) {
        if (g_children[i].pid > 0) kill(g_children[i].pid, SIGINT);
    }
    for (int i = 0; i < g_child_count; i++) {
        if (g_children[i].pid <= 0) continue;
        printf("  %s... ", g_children[i].name);
        fflush(stdout);
        int status;
        int waited = 0;
        for (int t = 0; t < 600; t++) {
            if (waitpid(g_children[i].pid, &status, WNOHANG) != 0) { waited = 1; break; }
            usleep(100000);
        }
        if (waited) {
            printf("ok\n");
        } else {
            kill(g_children[i].pid, SIGKILL);
            waitpid(g_children[i].pid, &status, 0);
            printf("killed\n");
        }
    }
}

static void deploy_sig_handler(int sig) {
    (void)sig;
    printf("\nshutting down deploy...\n");
    deploy_terminate_children();
    printf("deploy shutdown complete\n");
    _exit(0);
}

static int deploy_spawn_child(const char *exepath, deploy_entry *e) {
    deploy_child *c = &g_children[g_child_count];
    memset(c, 0, sizeof(*c));
    strncpy(c->name, e->name, sizeof(c->name) - 1);

    int stdout_pipe[2], stderr_pipe[2];
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        fprintf(stderr, "\n  ERROR: pipe() failed: %s\n", strerror(errno));
        return 0;
    }

    char dim_arg[64];
    if (e->format[0])
        snprintf(dim_arg, sizeof(dim_arg), "%d:%s", e->dim, e->format);
    else
        snprintf(dim_arg, sizeof(dim_arg), "%d", e->dim);

    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "\n  ERROR: fork() failed: %s\n", strerror(errno));
        return 0;
    }
    if (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execl(exepath, exepath, "--notcp", "--nomotd", e->name, dim_arg, (char *)NULL);
        _exit(127);
    }

    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    c->pid = pid;
    c->stdout_fd = stdout_pipe[0];
    c->stderr_fd = stderr_pipe[0];
    c->ready = 0;
    g_child_count++;

    pthread_t t1, t2;
    pthread_create(&t1, NULL, deploy_stdout_reader, c);
    pthread_detach(t1);
    pthread_create(&t2, NULL, deploy_stderr_reader, c);
    pthread_detach(t2);

    /* wait for ready */
    while (!c->ready) {
        int status;
        pid_t r = waitpid(c->pid, &status, WNOHANG);
        if (r > 0) {
            fprintf(stderr, "\n  ERROR: exited before ready\n");
            c->pid = 0;
            return 0;
        }
        usleep(50000);
    }
    return 1;
}

static int run_deploy(int port) {
    char exepath[512] = {0};
    ssize_t len = readlink("/proc/self/exe", exepath, sizeof(exepath) - 1);
    if (len <= 0) {
        fprintf(stderr, "ERROR: cannot determine own executable path\n");
        return 1;
    }
    exepath[len] = '\0';

    /* find longest name for alignment */
    int maxlen = 0;
    for (int i = 0; i < g_deploy_count; i++) {
        int l = (int)strlen(g_deploy[i].name);
        if (l > maxlen) maxlen = l;
    }

    printf("loading databases...\n");
    for (int i = 0; i < g_deploy_count; i++) {
        deploy_entry *e = &g_deploy[i];
        printf("  %s%*s", e->name, maxlen - (int)strlen(e->name) + 3, "");
        fflush(stdout);
        if (!deploy_spawn_child(exepath, e)) {
            deploy_terminate_children();
            return 1;
        }
        printf("ok\n");
    }

    /* print socket endpoints */
    for (int i = 0; i < g_deploy_count; i++)
        printf("Socket listening on /tmp/vec_%s.sock\n", g_deploy[i].name);

    struct sigaction sa;
    sa.sa_handler = deploy_sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    return run_router(port, 1);
}

/* ----- Signal + Main (Linux) ----------------------------------------- */

static void sig_handler(int sig) {
    (void)sig;
    if (!g_running) return;
    g_running = 0;
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
        if (strncmp(argv[i], "--deploy=", 9) == 0) {
            if (parse_deploy(argv[i] + 9) <= 0) return 1;
            int dport = DEFAULT_PORT;
            if (i + 1 < argc) { int p = atoi(argv[i + 1]); if (p > 0 && p <= 65535) dport = p; }
            return run_deploy(dport);
        }

        if (strcmp(argv[i], "deploy") == 0) {
            int add_mode = (i + 1 < argc && strcmp(argv[i + 1], "--add") == 0);
            int found = discover_all_dbs();
            if (add_mode) {
                deploy_wizard(1);
                if (g_deploy_count <= found) { fprintf(stderr, "no database added\n"); return 1; }
                for (int j = found; j < g_deploy_count; j++)
                    write_empty_tensors(&g_deploy[j]);
            } else if (found <= 0) {
                found = deploy_wizard(1);
                if (found <= 0) return 1;
                system("clear");
                found = g_deploy_count;
            }
            if (g_deploy_count == 1) {
                fprintf(stderr, "WARNING: deploy is for multiple databases, launching single instance\n");
                argc = 1; /* fall through to normal single-db startup */
                break;
            }
            int dport = DEFAULT_PORT;
            for (int j = i + 1; j < argc; j++) {
                if (strcmp(argv[j], "--add") == 0) continue;
                int p = atoi(argv[j]);
                if (p > 0 && p <= 65535) { dport = p; break; }
            }
            return run_deploy(dport);
        }

        if (strcmp(argv[i], "--delete") == 0) {
            if (i < 2 || strcmp(argv[1], "--delete") == 0) {
                fprintf(stderr, "usage: vec <name> --delete\n");
                return 1;
            }
            const char *name = argv[1];
            char pattern[512];
            int deleted = 0;

            snprintf(pattern, sizeof(pattern), "%s_*.tensors", name);
            glob_t gl;
            if (glob(pattern, 0, NULL, &gl) == 0) {
                for (size_t j = 0; j < gl.gl_pathc; j++) {
                    if (remove(gl.gl_pathv[j]) == 0) {
                        printf("deleted %s\n", gl.gl_pathv[j]);
                        deleted++;
                    } else {
                        fprintf(stderr, "ERROR: could not delete %s\n", gl.gl_pathv[j]);
                    }
                }
                globfree(&gl);
            }

            snprintf(pattern, sizeof(pattern), "%s_*.meta", name);
            if (glob(pattern, 0, NULL, &gl) == 0) {
                for (size_t j = 0; j < gl.gl_pathc; j++) {
                    if (remove(gl.gl_pathv[j]) == 0) {
                        printf("deleted %s\n", gl.gl_pathv[j]);
                        deleted++;
                    } else {
                        fprintf(stderr, "ERROR: could not delete %s\n", gl.gl_pathv[j]);
                    }
                }
                globfree(&gl);
            }

            if (deleted == 0) {
                fprintf(stderr, "no database files found for '%s'\n", name);
                return 1;
            }
            printf("destroyed %d file(s) for database '%s'\n", deleted, name);
            return 0;
        }

        if (strcmp(argv[i], "--check") == 0 || strcmp(argv[i], "--repair") == 0) {
            int dry_run = (strcmp(argv[i], "--check") == 0);
            if (i >= 2) {
                strncpy(g_name, argv[1], sizeof(g_name) - 1);
                if (!find_existing_db(argv[1])) {
                    fprintf(stderr, "ERROR: no database found for '%s'\n", argv[1]);
                    return 1;
                }
                return run_repair(dry_run);
            }
            glob_t cgl;
            if (glob("*.tensors", 0, NULL, &cgl) != 0) {
                fprintf(stderr, "no .tensors files found in current directory\n");
                return 1;
            }
            int total = 0, errors = 0;
            for (size_t j = 0; j < cgl.gl_pathc; j++) {
                strncpy(g_filepath, cgl.gl_pathv[j], sizeof(g_filepath) - 1);
                printf("--- %s ---\n", g_filepath);
                int r = run_repair(dry_run);
                if (r != 0) errors++;
                total++;
                printf("\n");
            }
            globfree(&cgl);
            printf("checked %d database(s), %d with issues\n", total, errors);
            return (errors > 0) ? 1 : 0;
        }
    }

    int no_tcp = 0;
    int no_motd = 0;
    if (argc > 1 && strcmp(argv[1], "--notcp") == 0) { no_tcp = 1; argc--; argv++; }
    if (argc > 1 && strcmp(argv[1], "--nomotd") == 0) { no_motd = 1; argc--; argv++; }

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

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    double vram_gb = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    double max_records = (prop.totalGlobalMem * 0.9) / ((double)g_dim * g_elem_size);

    if (!no_motd) printf("%s (%.1f GB)\n", prop.name, vram_gb);

    gpu_init();
    cudaGetLastError();

    launch_l2_f32((const float *)d_vectors, (const float *)d_query, d_dists, 1, g_dim);
    cudaDeviceSynchronize();
    cudaGetLastError();

    int lr = load_from_file();
    if (lr < 0) {
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }

    /* check write permission without creating the file */
    {
        FILE *tf = fopen(g_filepath, "r+b"); /* existing file: open for update */
        if (tf) { fclose(tf); }
        else if (errno == ENOENT) {
            /* file doesn't exist yet — probe directory writability via a temp file */
            char probe[520];
            snprintf(probe, sizeof(probe), "_wprb%u.tmp", (unsigned)time(NULL));
            FILE *pf = fopen(probe, "wb");
            if (pf) { fclose(pf); remove(probe); }
            else { g_readonly = 1; }
        } else { g_readonly = 1; }
    }

    if (!no_motd) print_startup_info(file_exists, lr, max_records);
    if (g_readonly) printf("WARNING: read-only mode (cannot write to %s)\n", g_filepath);

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
    if (no_motd) {
        putchar('\x06');
        fflush(stdout);
    } else {
        printf("ready for connections, ctrl+c to exit\n");
        if (no_tcp)
            printf("hint: use --route to expose this instance via TCP\n");
    }

    while (g_running) {
        usleep(500000);
        if (g_dirty && g_last_write && (time(NULL) - g_last_write >= AUTOSAVE_IDLE_SECS)) {
            REQ_LOCK();
            if (g_dirty && (time(NULL) - g_last_write >= AUTOSAVE_IDLE_SECS))
                save_to_file(1);
            REQ_UNLOCK();
        }
    }

    printf("\nshutting down...\n");
    if (has_tcp) pthread_join(t_tcp, NULL);
    pthread_join(t_unix, NULL);

    save_to_file(0);
    gpu_shutdown();
    release_instance_lock();
    return 0;
}

#endif /* _WIN32 */
