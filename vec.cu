/*
 * vec.cu — GPU-resident vector database
 *
 * Usage:  vec <name> <dim>
 *
 * Creates an in-memory CUDA vector store. Listens on:
 *   - TCP port 1920
 *   - Named pipe: \\.\pipe\vec_<name>
 *
 * Text protocol (TCP & pipe):
 *   push 1.0,2.0,3.0,...\n        -> returns slot index
 *   pull 1.0,2.0,3.0,...\n        -> returns top 10 nearest (index + distance)
 *   bpush <N>\n<N*dim*4 bytes>    -> binary bulk push, returns first slot index
 *   save\n                         -> flush to <name>.tensors
 *   count\n                        -> returns current vector count
 *   quit\n                         -> disconnect client
 *
 * File format (.tensors):
 *   [4B dim][4B count][count*dim*4B float data]
 *
 * Ctrl+C saves before exit.
 *
 * Build (Windows, Ampere+):
 *   nvcc -O3 vec.cu -o vec.exe -lws2_32 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
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

#define TCP_PORT        1920
#define TOP_K           10
#define INITIAL_CAP     4096
#define MAX_LINE        (1 << 20)   /* 1 MB max command line */
#define PIPE_BUF_SIZE   (1 << 16)

/* ------------------------------------------------------------------ */
/*  Globals                                                            */
/* ------------------------------------------------------------------ */

static char g_name[256];
static int  g_dim        = 0;
static int  g_count      = 0;
static int  g_capacity   = 0;

static float* d_vectors  = NULL;   /* GPU buffer [capacity * dim] */
static float* d_query    = NULL;   /* GPU buffer [dim]            */
static float* d_dists    = NULL;   /* GPU buffer [capacity]       */

/* host-side scratch */
static float* h_dists    = NULL;
static int*   h_ids      = NULL;
static float* h_pinned   = NULL;   /* pinned staging for transfers */
static int    h_pinned_n = 0;

static volatile int g_running = 1;

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
/*  GPU kernel: L2 distance (one thread per database vector)           */
/* ------------------------------------------------------------------ */

__global__ void kernel_l2_dist(const float* __restrict__ db,
                               const float* __restrict__ query,
                               float* __restrict__ dists,
                               int n, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* v = db + (size_t)i * dim;
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = v[d] - query[d];
        sum += diff * diff;
    }
    dists[i] = sum;
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

    /* reallocate distance buffer */
    if (d_dists) CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaMalloc(&d_dists, new_cap * sizeof(float)));

    /* host scratch */
    free(h_dists); free(h_ids);
    h_dists = (float*)malloc(new_cap * sizeof(float));
    h_ids   = (int*)malloc(new_cap * sizeof(int));

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

    /* pinned staging: start with room for 1024 vectors */
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
    free(h_dists); free(h_ids);
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

/*
 * Query top-K nearest by squared L2 distance.
 * Returns actual k (may be < TOP_K if count < TOP_K).
 */
static int vec_pull(const float* h_query, int* out_ids, float* out_dists)
{
    if (g_count == 0) return 0;
    int n = g_count;
    int k = (n < TOP_K) ? n : TOP_K;

    /* upload query */
    CUDA_CHECK(cudaMemcpy(d_query, h_query, g_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* compute L2 distances on GPU */
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    kernel_l2_dist<<<blocks, threads>>>(d_vectors, d_query, d_dists, n, g_dim);

    /* download distances */
    CUDA_CHECK(cudaMemcpy(h_dists, d_dists, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    /* partial selection sort for top-k on CPU */
    for (int i = 0; i < n; i++) h_ids[i] = i;

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
    char path[512];
    snprintf(path, sizeof(path), "%s.tensors", g_name);

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", path); return; }

    fwrite(&g_dim,   sizeof(int), 1, f);
    fwrite(&g_count, sizeof(int), 1, f);

    if (g_count > 0) {
        size_t total = (size_t)g_count * g_dim;
        gpu_ensure_pinned((int)total);
        CUDA_CHECK(cudaMemcpy(h_pinned, d_vectors, total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        fwrite(h_pinned, sizeof(float), total, f);
    }
    fclose(f);
    printf("[vec] saved %d vectors to %s\n", g_count, path);
}

static int load_from_file()
{
    char path[512];
    snprintf(path, sizeof(path), "%s.tensors", g_name);

    FILE* f = fopen(path, "rb");
    if (!f) return 0; /* no file, fresh start */

    int file_dim, file_count;
    if (fread(&file_dim, sizeof(int), 1, f) != 1 ||
        fread(&file_count, sizeof(int), 1, f) != 1) {
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
        size_t total = (size_t)file_count * g_dim;
        gpu_realloc_if_needed(file_count);
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
        g_count = file_count;
    }

    fclose(f);
    printf("[vec] loaded %d vectors from %s\n", g_count, path);
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

/* Process one text command. Returns: 0=ok, 1=quit */
static int process_command(const char* line, int line_len,
                           write_fn writer, void* wctx,
                           const char* bin_payload, int bin_payload_len)
{
    char resp[4096];
    int rlen;

    /* trim trailing whitespace */
    while (line_len > 0 && (line[line_len-1] == '\n' || line[line_len-1] == '\r'))
        line_len--;
    if (line_len == 0) return 0;

    /* --- push --- */
    if (line_len > 5 && strncmp(line, "push ", 5) == 0) {
        float* vals = (float*)malloc(g_dim * sizeof(float));
        int n = parse_floats(line + 5, vals, g_dim);
        if (n != g_dim) {
            rlen = snprintf(resp, sizeof(resp),
                            "ERR dim mismatch: got %d, expected %d\n", n, g_dim);
            writer(wctx, resp, rlen);
            free(vals);
            return 0;
        }
        int slot = vec_push(vals);
        free(vals);
        rlen = snprintf(resp, sizeof(resp), "%d\nOK\n", slot);
        writer(wctx, resp, rlen);
        return 0;
    }

    /* --- bpush <N> --- */
    if (line_len > 6 && strncmp(line, "bpush ", 6) == 0) {
        int n = atoi(line + 6);
        if (n <= 0) {
            rlen = snprintf(resp, sizeof(resp), "ERR invalid count\n");
            writer(wctx, resp, rlen);
            return 0;
        }
        int expected_bytes = n * g_dim * (int)sizeof(float);
        if (bin_payload_len < expected_bytes) {
            rlen = snprintf(resp, sizeof(resp), "ERR need %d bytes, got %d\n",
                            expected_bytes, bin_payload_len);
            writer(wctx, resp, rlen);
            return 0;
        }
        gpu_ensure_pinned(n * g_dim);
        memcpy(h_pinned, bin_payload, expected_bytes);
        int first = vec_bpush(n);
        rlen = snprintf(resp, sizeof(resp), "%d\nOK\n", first);
        writer(wctx, resp, rlen);
        return 0;
    }

    /* --- pull --- */
    if (line_len > 5 && strncmp(line, "pull ", 5) == 0) {
        float* vals = (float*)malloc(g_dim * sizeof(float));
        int n = parse_floats(line + 5, vals, g_dim);
        if (n != g_dim) {
            rlen = snprintf(resp, sizeof(resp),
                            "ERR dim mismatch: got %d, expected %d\n", n, g_dim);
            writer(wctx, resp, rlen);
            free(vals);
            return 0;
        }
        int ids[TOP_K];
        float dists[TOP_K];
        int k = vec_pull(vals, ids, dists);
        free(vals);

        char* p = resp;
        int rem = sizeof(resp);
        for (int i = 0; i < k; i++) {
            int w = snprintf(p, rem, "%d %.6f\n", ids[i], dists[i]);
            p += w; rem -= w;
        }
        int w = snprintf(p, rem, "OK\n");
        p += w;
        writer(wctx, resp, (int)(p - resp));
        return 0;
    }

    /* --- save --- */
    if (line_len >= 4 && strncmp(line, "save", 4) == 0) {
        save_to_file();
        rlen = snprintf(resp, sizeof(resp), "OK\n");
        writer(wctx, resp, rlen);
        return 0;
    }

    /* --- count --- */
    if (line_len >= 5 && strncmp(line, "count", 5) == 0) {
        rlen = snprintf(resp, sizeof(resp), "%d\nOK\n", g_count);
        writer(wctx, resp, rlen);
        return 0;
    }

    /* --- quit --- */
    if (line_len >= 4 && strncmp(line, "quit", 4) == 0) {
        rlen = snprintf(resp, sizeof(resp), "BYE\n");
        writer(wctx, resp, rlen);
        return 1;
    }

    rlen = snprintf(resp, sizeof(resp), "ERR unknown command\n");
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
    free(param); /* was heap-allocated by listener */

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
        return 1;
    }

    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(TCP_PORT);

    if (bind(listen_sock, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "ERROR: bind() failed: %d\n", WSAGetLastError());
        closesocket(listen_sock);
        return 1;
    }

    listen(listen_sock, SOMAXCONN);
    printf("[vec] TCP listening on 127.0.0.1:%d\n", TCP_PORT);

    while (g_running) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_sock, &fds);
        struct timeval tv;
        tv.tv_sec = 1; tv.tv_usec = 0;
        int sel = select(0, &fds, NULL, NULL, &tv);
        if (sel <= 0) continue;

        SOCKET client = accept(listen_sock, NULL, NULL);
        if (client == INVALID_SOCKET) continue;

        /* heap-allocate the SOCKET so the thread can read it safely */
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

    printf("[vec] Pipe listening on %s\n", pipe_name);

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

        /* wait for client with periodic g_running check */
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

        /* read commands from this pipe client */
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
        printf("\n[vec] shutting down...\n");
        save_to_file();
        g_running = 0;
        return TRUE;
    }
    return FALSE;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: vec <name> <dim>\n");
        fprintf(stderr, "  Creates a GPU vector database.\n");
        fprintf(stderr, "  Listens on TCP port %d and named pipe \\\\.\\pipe\\vec_<name>\n",
                TCP_PORT);
        return 1;
    }

    strncpy(g_name, argv[1], sizeof(g_name) - 1);
    g_dim = atoi(argv[2]);

    if (g_dim <= 0 || g_dim > 65536) {
        fprintf(stderr, "ERROR: dimension must be between 1 and 65536\n");
        return 1;
    }

    printf("[vec] name=%s dim=%d\n", g_name, g_dim);

    /* init GPU */
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("[vec] GPU: %s (%.0f MB)\n", prop.name,
           prop.totalGlobalMem / (1024.0 * 1024.0));

    gpu_init();

    /* auto-load if file exists */
    int lr = load_from_file();
    if (lr < 0) {
        gpu_shutdown();
        return 1;
    }

    /* ctrl+c handler */
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    /* start listeners */
    HANDLE h_tcp  = CreateThread(NULL, 0, tcp_listener_thread, NULL, 0, NULL);
    HANDLE h_pipe = CreateThread(NULL, 0, pipe_listener_thread, NULL, 0, NULL);

    printf("[vec] ready. %d vectors loaded. Ctrl+C to save & exit.\n", g_count);

    while (g_running) {
        Sleep(500);
    }

    WaitForSingleObject(h_tcp,  3000);
    WaitForSingleObject(h_pipe, 3000);

    gpu_shutdown();
    printf("[vec] done.\n");
    return 0;
}
