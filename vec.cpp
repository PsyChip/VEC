/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * vec - dead simple GPU-resident vector database (host code, fp32)
 *
 * Usage:  vec <name> <dim> [port]
 *
 * Creates an in-memory CUDA vector store.
 * Listens on:
 *   - TCP port (default 1920)
 *   - Named pipe: \\.\pipe\vec_<name>
 *
 * Protocol (TCP & pipe):
 *   push 1.0,2.0,3.0,...\n        -> returns slot index
 *   pull 1.0,2.0,3.0,...\n        -> top 10 nearest by L2 distance (index:distance,...)
 *   cpull 1.0,2.0,3.0,...\n       -> top 10 nearest by cosine distance
 *   bpush <N>\n<N*dim*4 bytes>    -> binary bulk push (fp32), returns first slot index
 *   delete <index>\n              -> tombstone a vector
 *   undo\n                         -> remove last pushed vector
 *   save\n                         -> flush to <name>.tensors
 *   size\n                         -> returns total index count
 *
 * File format (.tensors):
 *   [4B dim][4B count][4B deleted][count B alive mask][count*dim*4B fp32 data]
 *
 * Ctrl+C saves before exit.
 *
 * Build (Windows):
 *   nvcc -O2 vec_kernel.cu vec.cpp -o vec.exe -lws2_32 -arch=sm_86
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Config                                                             */
/* ------------------------------------------------------------------ */

#define DEFAULT_PORT    1920
#define TOP_K           10
#define INITIAL_CAP     4096
#define MAX_LINE        (1 << 20)
#define PIPE_BUF_SIZE   (1 << 16)

/* ------------------------------------------------------------------ */
/*  External kernels (vec_kernel.cu)                                   */
/* ------------------------------------------------------------------ */

extern "C" {
void launch_l2_dist(const float* db, const float* query,
                    float* dists, int n, int dim);
void launch_cos_dist(const float* db, const float* query,
                     float* dists, int n, int dim);
}

/* ------------------------------------------------------------------ */
/*  Globals                                                            */
/* ------------------------------------------------------------------ */

static char g_name[256];
static int  g_dim        = 0;
static int  g_port       = DEFAULT_PORT;
static int  g_count      = 0;
static int  g_capacity   = 0;

static float* d_vectors  = NULL;   /* GPU fp32 [capacity * dim] */
static float* d_query    = NULL;   /* GPU fp32 [dim]            */
static float* d_dists    = NULL;   /* GPU fp32 [capacity]       */

static float* h_dists    = NULL;
static int*   h_ids      = NULL;
static float* h_pinned   = NULL;   /* pinned host staging */
static int    h_pinned_n = 0;

static unsigned char* g_alive = NULL;
static int    g_alive_cap = 0;
static int    g_deleted   = 0;

static volatile int g_running = 1;
static HANDLE       g_mutex   = NULL;

/* ------------------------------------------------------------------ */
/*  CUDA helpers                                                       */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/*  Instance guard: named mutex                                        */
/* ------------------------------------------------------------------ */

static int acquire_instance_lock()
{
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

static void release_instance_lock()
{
    if (g_mutex) {
        ReleaseMutex(g_mutex);
        CloseHandle(g_mutex);
        g_mutex = NULL;
    }
}

/* ------------------------------------------------------------------ */
/*  GPU memory management                                              */
/* ------------------------------------------------------------------ */

static void gpu_realloc_if_needed(int required)
{
    if (required <= g_capacity) return;

    int new_cap = g_capacity;
    while (new_cap < required) new_cap *= 2;

    float* d_new;
    CUDA_CHECK(cudaMalloc(&d_new, (size_t)new_cap * g_dim * sizeof(float)));
    if (d_vectors && g_count > 0) {
        CUDA_CHECK(cudaMemcpy(d_new, d_vectors,
                              (size_t)g_count * g_dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    if (d_vectors) CUDA_CHECK(cudaFree(d_vectors));
    d_vectors = d_new;

    if (d_dists) CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaMalloc(&d_dists, new_cap * sizeof(float)));

    free(h_dists); free(h_ids);
    h_dists = (float*)malloc(new_cap * sizeof(float));
    h_ids   = (int*)malloc(new_cap * sizeof(int));

    unsigned char* new_alive = (unsigned char*)realloc(g_alive, new_cap);
    memset(new_alive + g_alive_cap, 1, new_cap - g_alive_cap);
    g_alive = new_alive;
    g_alive_cap = new_cap;

    g_capacity = new_cap;
}

static void gpu_init()
{
    g_capacity = INITIAL_CAP;
    CUDA_CHECK(cudaMalloc(&d_vectors, (size_t)g_capacity * g_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query,   g_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dists,   g_capacity * sizeof(float)));

    h_dists = (float*)malloc(g_capacity * sizeof(float));
    h_ids   = (int*)malloc(g_capacity * sizeof(int));

    g_alive = (unsigned char*)malloc(g_capacity);
    memset(g_alive, 1, g_capacity);
    g_alive_cap = g_capacity;

    h_pinned_n = 1024 * g_dim;
    CUDA_CHECK(cudaMallocHost(&h_pinned, h_pinned_n * sizeof(float)));
}

static void gpu_ensure_pinned(int nfloats)
{
    if (nfloats <= h_pinned_n) return;
    CUDA_CHECK(cudaFreeHost(h_pinned));
    h_pinned_n = nfloats;
    CUDA_CHECK(cudaMallocHost(&h_pinned, h_pinned_n * sizeof(float)));
}

static void gpu_shutdown()
{
    if (d_vectors) cudaFree(d_vectors);
    if (d_query)   cudaFree(d_query);
    if (d_dists)   cudaFree(d_dists);
    if (h_pinned)  cudaFreeHost(h_pinned);
    free(h_dists); free(h_ids); free(g_alive);
}

/* ------------------------------------------------------------------ */
/*  Vector operations                                                  */
/* ------------------------------------------------------------------ */

static int vec_push(const float* h_vec)
{
    gpu_realloc_if_needed(g_count + 1);
    int slot = g_count;
    CUDA_CHECK(cudaMemcpy(d_vectors + (size_t)slot * g_dim, h_vec,
                          g_dim * sizeof(float), cudaMemcpyHostToDevice));
    g_count++;
    return slot;
}

static int vec_bpush(int n)
{
    gpu_realloc_if_needed(g_count + n);
    int first = g_count;
    CUDA_CHECK(cudaMemcpy(d_vectors + (size_t)first * g_dim, h_pinned,
                          (size_t)n * g_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    g_count += n;
    return first;
}

/* mode: 0=L2, 1=cosine */
static int vec_pull(const float* h_query, int* out_ids, float* out_dists, int mode)
{
    int alive = g_count - g_deleted;
    if (alive <= 0) return 0;
    int n = g_count;
    int k = (alive < TOP_K) ? alive : TOP_K;

    cudaGetLastError(); /* clear any stale errors */

    CUDA_CHECK(cudaMemcpy(d_query, h_query, g_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    if (mode == 1)
        launch_cos_dist(d_vectors, d_query, d_dists, n, g_dim);
    else
        launch_l2_dist(d_vectors, d_query, d_dists, n, g_dim);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_dists, d_dists, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

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
        out_ids[i]   = h_ids[i];
        out_dists[i] = h_dists[i];
    }
    return k;
}

/* ------------------------------------------------------------------ */
/*  Persistence                                                        */
/* ------------------------------------------------------------------ */

static void save_to_file()
{
    if (g_count == 0) { printf("nothing to save\n"); return; }

    char path[512];
    snprintf(path, sizeof(path), "%s.tensors", g_name);

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", path); return; }

    fwrite(&g_dim,     sizeof(int), 1, f);
    fwrite(&g_count,   sizeof(int), 1, f);
    fwrite(&g_deleted, sizeof(int), 1, f);

    if (g_count > 0) {
        fwrite(g_alive, 1, g_count, f);

        size_t total = (size_t)g_count * g_dim;
        gpu_ensure_pinned((int)total);
        CUDA_CHECK(cudaMemcpy(h_pinned, d_vectors, total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        fwrite(h_pinned, sizeof(float), total, f);
    }
    fclose(f);
    printf("saved %d vectors (%d deleted) to %s\n", g_count, g_deleted, path);
}

static int load_from_file()
{
    char path[512];
    snprintf(path, sizeof(path), "%s.tensors", g_name);

    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    int file_dim, file_count, file_deleted;
    if (fread(&file_dim, sizeof(int), 1, f) != 1 ||
        fread(&file_count, sizeof(int), 1, f) != 1 ||
        fread(&file_deleted, sizeof(int), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "WARN: corrupt %s, starting fresh\n", path);
        return 0;
    }

    if (file_dim != g_dim) {
        fprintf(stderr, "ERROR: %s has dim=%d but requested dim=%d\n",
                path, file_dim, g_dim);
        fclose(f);
        return -1;
    }

    if (file_count > 0) {
        gpu_realloc_if_needed(file_count);

        size_t mask_rd = fread(g_alive, 1, file_count, f);
        if ((int)mask_rd != file_count) {
            fprintf(stderr, "WARN: alive mask truncated\n");
            file_count = (int)mask_rd;
        }

        size_t total = (size_t)file_count * g_dim;
        gpu_ensure_pinned((int)total);

        size_t rd = fread(h_pinned, sizeof(float), total, f);
        if (rd != total) {
            fprintf(stderr, "WARN: expected %llu floats, got %llu\n",
                    (unsigned long long)total, (unsigned long long)rd);
            file_count = (int)(rd / g_dim);
        }

        CUDA_CHECK(cudaMemcpy(d_vectors, h_pinned,
                              (size_t)file_count * g_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        g_count   = file_count;
        g_deleted = file_deleted;
    }

    fclose(f);
    printf("loaded %d vectors (%d deleted) from %s\n", g_count, g_deleted, path);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Protocol parser                                                    */
/* ------------------------------------------------------------------ */

static int parse_floats(const char* s, float* out, int max_n)
{
    int n = 0;
    const char* p = s;
    while (*p && n < max_n) {
        char* end;
        float v = strtof(p, &end);
        if (end == p) break;
        out[n++] = v;
        p = end;
        if (*p == ',') p++;
    }
    return n;
}

typedef int (*write_fn)(void* ctx, const char* buf, int len);

static int process_command(const char* line, int line_len,
                           write_fn writer, void* wctx,
                           const char* bin_payload, int bin_payload_len)
{
    char resp[4096];
    int rlen;

    while (line_len > 0 && (line[line_len-1] == '\n' || line[line_len-1] == '\r'))
        line_len--;
    if (line_len == 0) return 0;

    if (line_len > 5 && strncmp(line, "push ", 5) == 0) {
        float* vals = (float*)malloc(g_dim * sizeof(float));
        int n = parse_floats(line + 5, vals, g_dim);
        if (n != g_dim) {
            rlen = snprintf(resp, sizeof(resp),
                            "err dim mismatch: got %d, expected %d\n", n, g_dim);
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
            rlen = snprintf(resp, sizeof(resp), "err need %d bytes, got %d\n",
                            expected_bytes, bin_payload_len);
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

    /* pull (L2) or cpull (cosine) */
    int pull_mode = -1;
    int pull_offset = 0;
    if (line_len > 5 && strncmp(line, "pull ", 5) == 0) {
        pull_mode = 0; pull_offset = 5;
    } else if (line_len > 6 && strncmp(line, "cpull ", 6) == 0) {
        pull_mode = 1; pull_offset = 6;
    }
    if (pull_mode >= 0) {
        float* vals = (float*)malloc(g_dim * sizeof(float));
        int n = parse_floats(line + pull_offset, vals, g_dim);
        if (n != g_dim) {
            rlen = snprintf(resp, sizeof(resp),
                            "err dim mismatch: got %d, expected %d\n", n, g_dim);
            writer(wctx, resp, rlen);
            free(vals);
            return 0;
        }
        int ids[TOP_K];
        float dists[TOP_K];
        int k = vec_pull(vals, ids, dists, pull_mode);
        free(vals);

        char* p = resp;
        int rem = sizeof(resp);
        for (int i = 0; i < k; i++) {
            int w = snprintf(p, rem, "%s%d:%.6f", i > 0 ? "," : "", ids[i], dists[i]);
            p += w; rem -= w;
        }
        *p++ = '\n'; rem--;
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

/* ------------------------------------------------------------------ */
/*  I/O: TCP                                                           */
/* ------------------------------------------------------------------ */

static int tcp_writer(void* ctx, const char* buf, int len)
{
    SOCKET s = *(SOCKET*)ctx;
    return send(s, buf, len, 0);
}

static DWORD WINAPI tcp_client_thread(LPVOID param)
{
    SOCKET client = *(SOCKET*)param;
    free(param);

    char* buf = (char*)malloc(MAX_LINE);
    int buf_used = 0;

    while (g_running) {
        int r = recv(client, buf + buf_used, MAX_LINE - buf_used - 1, 0);
        if (r <= 0) break;
        buf_used += r;
        buf[buf_used] = '\0';

        while (1) {
            char* nl = (char*)memchr(buf, '\n', buf_used);
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

                int rc = process_command(buf, line_len, tcp_writer, &client,
                                         buf + header_bytes, payload_bytes);
                int consumed = total_needed;
                buf_used -= consumed;
                if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                if (rc == 1) goto done;
            } else {
                int rc = process_command(buf, line_len, tcp_writer, &client,
                                         NULL, 0);
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

static DWORD WINAPI tcp_listener_thread(LPVOID param)
{
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
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons((unsigned short)g_port);

    if (bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        int err = WSAGetLastError();
        if (err == WSAEADDRINUSE) {
            fprintf(stderr, "ERROR: port %d is already in use\n", g_port);
        } else {
            fprintf(stderr, "ERROR: bind() failed: %d\n", err);
        }
        closesocket(listen_sock);
        g_running = 0;
        return 1;
    }

    listen(listen_sock, SOMAXCONN);
    printf("TCP listening on 127.0.0.1:%d\n", g_port);

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

        SOCKET* ps = (SOCKET*)malloc(sizeof(SOCKET));
        *ps = client;
        CreateThread(NULL, 0, tcp_client_thread, ps, 0, NULL);
    }

    closesocket(listen_sock);
    WSACleanup();
    return 0;
}

/* ------------------------------------------------------------------ */
/*  I/O: Named pipe                                                    */
/* ------------------------------------------------------------------ */

static int pipe_writer(void* ctx, const char* buf, int len)
{
    HANDLE pipe = *(HANDLE*)ctx;
    DWORD written;
    WriteFile(pipe, buf, len, &written, NULL);
    FlushFileBuffers(pipe);
    return (int)written;
}

static DWORD WINAPI pipe_listener_thread(LPVOID param)
{
    (void)param;
    char pipe_name[512];
    snprintf(pipe_name, sizeof(pipe_name), "\\\\.\\pipe\\vec_%s", g_name);

    printf("Pipe listening on %s\n", pipe_name);

    while (g_running) {
        HANDLE pipe = CreateNamedPipeA(
            pipe_name,
            PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES,
            PIPE_BUF_SIZE, PIPE_BUF_SIZE,
            1000,
            NULL);

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

        char* buf = (char*)malloc(MAX_LINE);
        int buf_used = 0;

        while (g_running) {
            DWORD bytesRead;
            BOOL ok = ReadFile(pipe, buf + buf_used, MAX_LINE - buf_used - 1,
                               &bytesRead, NULL);
            if (!ok || bytesRead == 0) break;
            buf_used += (int)bytesRead;
            buf[buf_used] = '\0';

            while (1) {
                char* nl = (char*)memchr(buf, '\n', buf_used);
                if (!nl) break;
                int line_len = (int)(nl - buf);

                if (line_len > 6 && strncmp(buf, "bpush ", 6) == 0) {
                    int n = atoi(buf + 6);
                    int payload_bytes = n * g_dim * (int)sizeof(float);
                    int header_bytes = line_len + 1;
                    int total_needed = header_bytes + payload_bytes;

                    while (buf_used < total_needed && g_running) {
                        ok = ReadFile(pipe, buf + buf_used,
                                      MAX_LINE - buf_used, &bytesRead, NULL);
                        if (!ok || bytesRead == 0) goto pipe_client_done;
                        buf_used += (int)bytesRead;
                    }

                    int rc = process_command(buf, line_len, pipe_writer, &pipe,
                                             buf + header_bytes, payload_bytes);
                    int consumed = total_needed;
                    buf_used -= consumed;
                    if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                    if (rc == 1) goto pipe_client_done;
                } else {
                    int rc = process_command(buf, line_len, pipe_writer, &pipe,
                                             NULL, 0);
                    int consumed = line_len + 1;
                    buf_used -= consumed;
                    if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                    if (rc == 1) goto pipe_client_done;
                }
            }
        }

pipe_client_done:
        free(buf);
        FlushFileBuffers(pipe);
        DisconnectNamedPipe(pipe);
        CloseHandle(pipe);
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Signal handling                                                    */
/* ------------------------------------------------------------------ */

static BOOL WINAPI ctrl_handler(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        printf("\nshutting down...\n");
        save_to_file();
        g_running = 0;
        return TRUE;
    }
    return FALSE;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

static void generate_random_name(char* buf, int len)
{
    const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    srand((unsigned)GetTickCount());
    for (int i = 0; i < len; i++)
        buf[i] = chars[rand() % (sizeof(chars) - 1)];
    buf[len] = '\0';
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        generate_random_name(g_name, 6);
        g_dim = 1024;
    } else if (argc < 3) {
        strncpy(g_name, argv[1], sizeof(g_name) - 1);
        g_dim = 1024;
    } else {
        strncpy(g_name, argv[1], sizeof(g_name) - 1);
        g_dim = atoi(argv[2]);
    }

    if (argc >= 4) {
        g_port = atoi(argv[3]);
        if (g_port <= 0 || g_port > 65535) {
            fprintf(stderr, "ERROR: port must be between 1 and 65535\n");
            return 1;
        }
    }

    if (g_dim <= 0 || g_dim > 65536) {
        fprintf(stderr, "ERROR: dimension must be between 1 and 65536\n");
        return 1;
    }

    if (!acquire_instance_lock()) {
        return 1;
    }

    printf("name=%s dim=%d port=%d storage=fp32\n", g_name, g_dim, g_port);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (%.0f MB)\n", prop.name,
           prop.totalGlobalMem / (1024.0 * 1024.0));

    gpu_init();
    cudaGetLastError(); /* clear any init-time errors */

    int lr = load_from_file();
    if (lr < 0) {
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }

    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    HANDLE h_tcp  = CreateThread(NULL, 0, tcp_listener_thread, NULL, 0, NULL);
    HANDLE h_pipe = CreateThread(NULL, 0, pipe_listener_thread, NULL, 0, NULL);

    Sleep(200);
    if (!g_running) {
        WaitForSingleObject(h_tcp, 2000);
        gpu_shutdown();
        release_instance_lock();
        return 1;
    }

    int alive = g_count - g_deleted;
    printf("ready. %d vectors loaded (%d active). Ctrl+C to save & exit.\n",
           g_count, alive);

    while (g_running) {
        Sleep(500);
    }

    WaitForSingleObject(h_tcp,  3000);
    WaitForSingleObject(h_pipe, 3000);

    gpu_shutdown();
    release_instance_lock();
    printf("done.\n");
    return 0;
}
