/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * vec - dead simple GPU-resident vector database (Linux host code)
 *
 * Usage:  vec <name> <dim[:format]> [port]
 *         format: f32 (default), f16
 *
 * Creates an in-memory CUDA vector store.
 * Listens on:
 *   - TCP port (default 1920)
 *   - Unix domain socket: /tmp/vec_<name>.sock
 *
 * Protocol (TCP & socket):
 *   push 1.0,2.0,3.0,...\n        -> returns slot index
 *   pull 1.0,2.0,3.0,...\n        -> top 10 nearest by L2 distance
 *   cpull 1.0,2.0,3.0,...\n       -> top 10 nearest by cosine distance
 *   bpush <N>\n<N*dim*4 bytes>    -> binary bulk push (fp32), returns first slot index
 *   delete <index>\n              -> tombstone a vector
 *   undo\n                        -> remove last pushed vector
 *   save\n                        -> flush to <name>.tensors
 *   size\n                        -> returns total index count
 *
 * File format (.tensors):
 *   [4B dim][4B count][4B deleted][1B format][count B alive mask][vector data]
 *
 * Ctrl+C saves before exit.
 *
 * Build:
 *   nvcc -O2 -c vec_kernel.cu -o vec_kernel.o -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
 *   nvcc -O2 vec_kernel.o vec_linux.cpp -o vec -lpthread
 */

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
#include <time.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define DEFAULT_PORT 1920
#define TOP_K 10
#define INITIAL_CAP 4096
#define MAX_LINE (1 << 24)
#define SOCK_BUF_SIZE (1 << 16)

#define FMT_F32 0
#define FMT_F16 1

extern "C" {
    void launch_l2_f32(const float *db, const float *query, float *dists, int n, int dim);
    void launch_cos_f32(const float *db, const float *query, float *dists, int n, int dim);
    void launch_l2_f16(const void *db, const void *query, float *dists, int n, int dim);
    void launch_cos_f16(const void *db, const void *query, float *dists, int n, int dim);
    void launch_f32_to_f16(const float *src, void *dst, int count);
}

static char g_name[256];
static char g_filepath[512];
static char g_sockpath[512];
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
static float *d_staging = NULL;
static int d_staging_n = 0;

static float *h_dists = NULL;
static int *h_ids = NULL;
static float *h_pinned = NULL;
static int h_pinned_n = 0;

static unsigned char *g_alive = NULL;
static int g_alive_cap = 0;
static int g_deleted = 0;

static volatile int g_running = 1;
static int g_lockfd = -1;

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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
    h_ids = (int *)malloc(new_cap * sizeof(int));

    unsigned char *new_alive = (unsigned char *)realloc(g_alive, new_cap);
    memset(new_alive + g_alive_cap, 1, new_cap - g_alive_cap);
    g_alive = new_alive;
    g_alive_cap = new_cap;
    g_capacity = new_cap;
}

static void gpu_init() {
    g_capacity = INITIAL_CAP;
    CUDA_CHECK(cudaMalloc(&d_vectors, (size_t)g_capacity * g_dim * g_elem_size));
    CUDA_CHECK(cudaMalloc(&d_query, g_dim * g_elem_size));
    CUDA_CHECK(cudaMalloc(&d_dists, g_capacity * sizeof(float)));

    if (g_fmt == FMT_F16) {
        d_staging_n = 1024 * g_dim;
        CUDA_CHECK(cudaMalloc(&d_staging, d_staging_n * sizeof(float)));
    }

    h_dists = (float *)malloc(g_capacity * sizeof(float));
    h_ids = (int *)malloc(g_capacity * sizeof(int));
    g_alive = (unsigned char *)malloc(g_capacity);
    memset(g_alive, 1, g_capacity);
    g_alive_cap = g_capacity;
    h_pinned_n = 1024 * g_dim;
    CUDA_CHECK(cudaMallocHost(&h_pinned, h_pinned_n * sizeof(float)));
}

static void gpu_ensure_pinned(int nfloats) {
    if (nfloats <= h_pinned_n) return;
    CUDA_CHECK(cudaFreeHost(h_pinned));
    h_pinned_n = nfloats;
    CUDA_CHECK(cudaMallocHost(&h_pinned, h_pinned_n * sizeof(float)));
}

static void gpu_shutdown() {
    if (d_vectors) cudaFree(d_vectors);
    if (d_query) cudaFree(d_query);
    if (d_dists) cudaFree(d_dists);
    if (d_staging) cudaFree(d_staging);
    if (h_pinned) cudaFreeHost(h_pinned);
    free(h_dists);
    free(h_ids);
    free(g_alive);
}

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

static int vec_push(const float *h_vec) {
    gpu_realloc_if_needed(g_count + 1);
    int slot = g_count;
    upload_and_store(h_vec, (char *)d_vectors + (size_t)slot * g_dim * g_elem_size, g_dim);
    g_count++;
    return slot;
}

static int vec_bpush(int n) {
    gpu_realloc_if_needed(g_count + n);
    int first = g_count;
    upload_and_store(h_pinned, (char *)d_vectors + (size_t)first * g_dim * g_elem_size, n * g_dim);
    g_count += n;
    return first;
}

static int vec_pull(const float *h_query, int *out_ids, float *out_dists, int mode) {
    int alive = g_count - g_deleted;
    if (alive <= 0) return 0;
    int n = g_count;
    int k = (alive < TOP_K) ? alive : TOP_K;

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

    for (int i = 0; i < k; i++) {
        out_ids[i] = h_ids[i];
        out_dists[i] = h_dists[i];
    }
    return k;
}

static void save_to_file() {
    if (g_count == 0) return;
    FILE *f = fopen(g_filepath, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", g_filepath); return; }

    fwrite(&g_dim, sizeof(int), 1, f);
    fwrite(&g_count, sizeof(int), 1, f);
    fwrite(&g_deleted, sizeof(int), 1, f);
    unsigned char fmt_byte = (unsigned char)g_fmt;
    fwrite(&fmt_byte, 1, 1, f);

    if (g_count > 0) {
        fwrite(g_alive, 1, g_count, f);
        size_t total_bytes = (size_t)g_count * g_dim * g_elem_size;
        void *h_buf = malloc(total_bytes);
        CUDA_CHECK(cudaMemcpy(h_buf, d_vectors, total_bytes, cudaMemcpyDeviceToHost));
        fwrite(h_buf, 1, total_bytes, f);
        free(h_buf);
    }
    fclose(f);
    printf("saved %d vectors to %s\n", g_count, g_filepath);
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

    if (file_count > 0) {
        gpu_realloc_if_needed(file_count);
        size_t mask_rd = fread(g_alive, 1, file_count, f);
        if ((int)mask_rd != file_count) {
            fprintf(stderr, "WARN: alive mask truncated\n");
            file_count = (int)mask_rd;
        }
        size_t total_bytes = (size_t)file_count * g_dim * g_elem_size;
        void *h_buf = malloc(total_bytes);
        size_t rd = fread(h_buf, 1, total_bytes, f);
        if (rd != total_bytes) {
            fprintf(stderr, "WARN: data truncated\n");
            file_count = (int)(rd / (g_dim * g_elem_size));
        }
        CUDA_CHECK(cudaMemcpy(d_vectors, h_buf, (size_t)file_count * g_dim * g_elem_size, cudaMemcpyHostToDevice));
        free(h_buf);
        g_count = file_count;
        g_deleted = file_deleted;
    }

    fclose(f);
    return 1;
}

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

static int process_command(const char *line, int line_len, write_fn writer, void *wctx, const char *bin_payload, int bin_payload_len) {
    char resp[4096];
    int rlen;

    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) line_len--;
    if (line_len == 0) return 0;

    if (line_len > 5 && strncmp(line, "push ", 5) == 0) {
        float *vals = (float *)malloc(g_dim * sizeof(float));
        int n = parse_floats(line + 5, vals, g_dim);
        if (n != g_dim) {
            rlen = snprintf(resp, sizeof(resp), "err dim mismatch: got %d, expected %d\n", n, g_dim);
            writer(wctx, resp, rlen);
            free(vals);
            return 0;
        }
        int slot = vec_push(vals);
        free(vals);
        rlen = snprintf(resp, sizeof(resp), "%d\n", slot);
        writer(wctx, resp, rlen);
        return 0;
    }

    if (line_len > 6 && strncmp(line, "bpush ", 6) == 0) {
        int n = atoi(line + 6);
        if (n <= 0) {
            rlen = snprintf(resp, sizeof(resp), "err invalid count\n");
            writer(wctx, resp, rlen);
            return 0;
        }
        int expected_bytes = n * g_dim * (int)sizeof(float);
        if (bin_payload_len < expected_bytes) {
            rlen = snprintf(resp, sizeof(resp), "err need %d bytes, got %d\n", expected_bytes, bin_payload_len);
            writer(wctx, resp, rlen);
            return 0;
        }
        gpu_ensure_pinned(n * g_dim);
        memcpy(h_pinned, bin_payload, expected_bytes);
        int first = vec_bpush(n);
        rlen = snprintf(resp, sizeof(resp), "%d\n", first);
        writer(wctx, resp, rlen);
        return 0;
    }

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
        char *p = resp;
        int rem = sizeof(resp);
        for (int i = 0; i < k; i++) {
            int w = snprintf(p, rem, "%s%d:%.6f", i > 0 ? "," : "", ids[i], dists[i]);
            p += w; rem -= w;
        }
        *p++ = '\n';
        writer(wctx, resp, (int)(p - resp));
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

            if (line_len > 6 && strncmp(buf, "bpush ", 6) == 0) {
                int n = atoi(buf + 6);
                int payload_bytes = n * g_dim * (int)sizeof(float);
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

static void sig_handler(int sig) {
    (void)sig;
    if (!g_running) return;
    g_running = 0;
    printf("\nshutting down...\n");
    save_to_file();
}

static void generate_random_name(char *buf, int len) {
    srand((unsigned)time(NULL));
    const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    for (int i = 0; i < len; i++) buf[i] = chars[rand() % (sizeof(chars) - 1)];
    buf[len] = '\0';
}

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

static int find_existing_db(const char *name) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s_*.tensors", name);

    glob_t gl;
    if (glob(pattern, 0, NULL, &gl) != 0) return 0;

    char best[512] = {0};
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

int main(int argc, char **argv) {
    int port_arg_idx = -1;

    if (argc < 2) {
        generate_random_name(g_name, 6);
        g_dim = 1024;
        build_filepath();
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

    printf("%s (%.1f GB)\n", prop.name, vram_gb);

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

    if (file_exists && lr > 0) {
        double remaining = max_records - g_count;
        double avail = (remaining / max_records) * 100.0;
        if (avail < 0) { avail = 0; remaining = 0; }
        if (g_count >= 1000000) printf("loading %.1fm records from %s\n", g_count / 1000000.0, g_filepath);
        else if (g_count >= 1000) printf("loading %.1fk records from %s\n", g_count / 1000.0, g_filepath);
        else printf("loading %d records from %s\n", g_count, g_filepath);
        if (remaining >= 1000000.0) printf("remaining space: %.1fm records (%.1f%%)\n", remaining / 1000000.0, avail);
        else if (remaining >= 1000.0) printf("remaining space: %.1fk records (%.1f%%)\n", remaining / 1000.0, avail);
        else printf("remaining space: %.0f records (%.1f%%)\n", remaining, avail);
    } else {
        printf("initializing database %s\n", g_filepath);
        if (max_records >= 1000000.0)
            printf("approx capacity: %.1fm records\n", max_records / 1000000.0);
        else if (max_records >= 1000.0)
            printf("approx capacity: %.1fk records\n", max_records / 1000.0);
        else
            printf("approx capacity: %.0f records\n", max_records);
    }

    struct sigaction sa;
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    pthread_t t_tcp, t_unix;
    pthread_create(&t_tcp, NULL, tcp_listener_thread, NULL);
    pthread_create(&t_unix, NULL, unix_listener_thread, NULL);

    usleep(200000);
    if (!g_running) {
        pthread_join(t_tcp, NULL);
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }
    printf("ready for connections, ctrl+c to exit\n");

    while (g_running) { usleep(500000); }

    pthread_join(t_tcp, NULL);
    pthread_join(t_unix, NULL);

    gpu_shutdown();
    release_instance_lock();
    return 0;
}
