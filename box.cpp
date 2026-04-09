/*
 * Curated by @PsyChip
 * root@psychip.net
 * April 2026
 *
 * box - dead simple memory-resident string store
 *
 * Usage:  box <name> [slot_size] [port]
 *
 * Allocates fixed-size UTF-8 string slots in system RAM.
 * Companion to vec - same index space, same append-only model.
 *
 * Listens on:
 *   - TCP port (default 2020)
 *   - Named pipe: \\.\pipe\box_<name>
 *
 * Protocol (TCP & pipe):
 *   push <string>\n               -> returns slot index
 *   pull <index>\n                -> returns string
 *   delete <index>\n              -> tombstone a slot
 *   undo\n                        -> remove last pushed slot
 *   save\n                        -> force flush to disk
 *   size\n                        -> returns total index count
 *
 * File format (.mem):
 *   [4B slot_size][4B count][4B deleted][count B alive mask][count * slot_size B string data]
 *
 * Ctrl+C flushes before exit.
 *
 * Build:
 *   cl /O2 /EHsc box.cpp /Fe:box.exe ws2_32.lib
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_PORT      2020
#define DEFAULT_SLOT_SIZE 255
#define INITIAL_CAP       4096
#define MAX_LINE          (1 << 20)
#define PIPE_BUF_SIZE     (1 << 16)

static char g_name[256];
static int g_slot_size = DEFAULT_SLOT_SIZE;
static int g_port = DEFAULT_PORT;
static int g_count = 0;
static int g_capacity = 0;
static int g_deleted = 0;

static char *g_data = NULL;
static unsigned char *g_alive = NULL;
static int g_alive_cap = 0;

static volatile int g_running = 1;
static HANDLE g_mutex = NULL;

static int acquire_instance_lock() {
    char mutex_name[512];
    snprintf(mutex_name, sizeof(mutex_name), "Global\\box_%s", g_name);
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

static void mem_realloc_if_needed(int required) {
    if (required <= g_capacity) return;
    int new_cap = g_capacity;
    while (new_cap < required) new_cap *= 2;

    g_data = (char *)realloc(g_data, (size_t)new_cap * g_slot_size);
    memset(g_data + (size_t)g_capacity * g_slot_size, 0, (size_t)(new_cap - g_capacity) * g_slot_size);

    unsigned char *new_alive = (unsigned char *)realloc(g_alive, new_cap);
    memset(new_alive + g_alive_cap, 1, new_cap - g_alive_cap);
    g_alive = new_alive;
    g_alive_cap = new_cap;
    g_capacity = new_cap;
}

static void mem_init() {
    g_capacity = INITIAL_CAP;
    g_data = (char *)calloc(g_capacity, g_slot_size);
    g_alive = (unsigned char *)malloc(g_capacity);
    memset(g_alive, 1, g_capacity);
    g_alive_cap = g_capacity;
}

static void mem_shutdown() {
    free(g_data);
    free(g_alive);
}

static int box_push(const char *str, int len) {
    mem_realloc_if_needed(g_count + 1);
    int slot = g_count;
    char *dst = g_data + (size_t)slot * g_slot_size;
    int copy_len = (len < g_slot_size) ? len : g_slot_size - 1;
    memcpy(dst, str, copy_len);
    dst[copy_len] = '\0';
    g_count++;
    return slot;
}

static const char *box_pull(int index) {
    if (index < 0 || index >= g_count) return NULL;
    if (!g_alive[index]) return NULL;
    return g_data + (size_t)index * g_slot_size;
}

static void save_to_file() {
    if (g_count == 0) { printf("nothing to save\n"); return; }
    char path[512];
    snprintf(path, sizeof(path), "%s.mem", g_name);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", path); return; }

    fwrite(&g_slot_size, sizeof(int), 1, f);
    fwrite(&g_count, sizeof(int), 1, f);
    fwrite(&g_deleted, sizeof(int), 1, f);

    if (g_count > 0) {
        fwrite(g_alive, 1, g_count, f);
        fwrite(g_data, g_slot_size, g_count, f);
    }
    fclose(f);
    printf("saved %d slots (%d deleted) to %s\n", g_count, g_deleted, path);
}

static int load_from_file() {
    char path[512];
    snprintf(path, sizeof(path), "%s.mem", g_name);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    int file_slot_size, file_count, file_deleted;
    if (fread(&file_slot_size, sizeof(int), 1, f) != 1 ||
        fread(&file_count, sizeof(int), 1, f) != 1 ||
        fread(&file_deleted, sizeof(int), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "WARN: corrupt %s, starting fresh\n", path);
        return 0;
    }

    if (file_slot_size != g_slot_size) {
        fprintf(stderr, "ERROR: %s has slot_size=%d but requested slot_size=%d\n", path, file_slot_size, g_slot_size);
        fclose(f);
        return -1;
    }

    if (file_count > 0) {
        mem_realloc_if_needed(file_count);
        size_t mask_rd = fread(g_alive, 1, file_count, f);
        if ((int)mask_rd != file_count) {
            fprintf(stderr, "WARN: alive mask truncated\n");
            file_count = (int)mask_rd;
        }
        size_t data_rd = fread(g_data, g_slot_size, file_count, f);
        if ((int)data_rd != file_count) {
            fprintf(stderr, "WARN: expected %d slots, got %d\n", file_count, (int)data_rd);
            file_count = (int)data_rd;
        }
        g_count = file_count;
        g_deleted = file_deleted;
    }

    fclose(f);
    printf("loaded %d slots (%d deleted) from %s\n", g_count, g_deleted, path);
    return 1;
}

typedef int (*write_fn)(void *ctx, const char *buf, int len);

static int process_command(const char *line, int line_len, write_fn writer, void *wctx) {
    char resp[4096];
    int rlen;

    while (line_len > 0 && (line[line_len - 1] == '\n' || line[line_len - 1] == '\r')) line_len--;
    if (line_len == 0) return 0;

    if (line_len > 5 && strncmp(line, "push ", 5) == 0) {
        const char *str = line + 5;
        int str_len = line_len - 5;
        int slot = box_push(str, str_len);
        rlen = snprintf(resp, sizeof(resp), "%d\n", slot);
        writer(wctx, resp, rlen);
        return 0;
    }

    if (line_len > 5 && strncmp(line, "pull ", 5) == 0) {
        int idx = atoi(line + 5);
        const char *val = box_pull(idx);
        if (!val) {
            if (idx >= 0 && idx < g_count && !g_alive[idx])
                rlen = snprintf(resp, sizeof(resp), "err deleted\n");
            else
                rlen = snprintf(resp, sizeof(resp), "err not found\n");
            writer(wctx, resp, rlen);
            return 0;
        }
        rlen = snprintf(resp, sizeof(resp), "%s\n", val);
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
        memset(g_data + (size_t)g_count * g_slot_size, 0, g_slot_size);
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
            int rc = process_command(buf, line_len, tcp_writer, &client);
            int consumed = line_len + 1;
            buf_used -= consumed;
            if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
            if (rc == 1) goto done;
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
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
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

        SOCKET *ps = (SOCKET *)malloc(sizeof(SOCKET));
        *ps = client;
        CreateThread(NULL, 0, tcp_client_thread, ps, 0, NULL);
    }

    closesocket(listen_sock);
    WSACleanup();
    return 0;
}

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
    snprintf(pipe_name, sizeof(pipe_name), "\\\\.\\pipe\\box_%s", g_name);
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
                int rc = process_command(buf, line_len, pipe_writer, &pipe);
                int consumed = line_len + 1;
                buf_used -= consumed;
                if (buf_used > 0) memmove(buf, buf + consumed, buf_used);
                if (rc == 1) goto pipe_done;
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

static BOOL WINAPI ctrl_handler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        printf("\nshutting down...\n");
        save_to_file();
        g_running = 0;
        return TRUE;
    }
    return FALSE;
}

static void generate_random_name(char *buf, int len) {
    const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    srand((unsigned)GetTickCount());
    for (int i = 0; i < len; i++) buf[i] = chars[rand() % (sizeof(chars) - 1)];
    buf[len] = '\0';
}

int main(int argc, char **argv) {
    if (argc < 2) {
        generate_random_name(g_name, 6);
        g_slot_size = DEFAULT_SLOT_SIZE;
    } else if (argc < 3) {
        strncpy(g_name, argv[1], sizeof(g_name) - 1);
        g_slot_size = DEFAULT_SLOT_SIZE;
    } else {
        strncpy(g_name, argv[1], sizeof(g_name) - 1);
        g_slot_size = atoi(argv[2]);
    }

    if (argc >= 4) {
        g_port = atoi(argv[3]);
        if (g_port <= 0 || g_port > 65535) {
            fprintf(stderr, "ERROR: port must be between 1 and 65535\n");
            return 1;
        }
    }

    if (g_slot_size <= 0 || g_slot_size > 65536) {
        fprintf(stderr, "ERROR: slot size must be between 1 and 65536\n");
        return 1;
    }

    if (!acquire_instance_lock()) return 1;

    printf("name=%s slot_size=%d port=%d\n", g_name, g_slot_size, g_port);

    mem_init();

    int lr = load_from_file();
    if (lr < 0) {
        mem_shutdown();
        release_instance_lock();
        return 1;
    }

    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    HANDLE h_tcp = CreateThread(NULL, 0, tcp_listener_thread, NULL, 0, NULL);
    HANDLE h_pipe = CreateThread(NULL, 0, pipe_listener_thread, NULL, 0, NULL);

    Sleep(200);
    if (!g_running) {
        WaitForSingleObject(h_tcp, 2000);
        mem_shutdown();
        release_instance_lock();
        return 1;
    }

    int alive = g_count - g_deleted;
    printf("ready. %d slots loaded (%d active). Ctrl+C to save & exit.\n", g_count, alive);

    while (g_running) { Sleep(500); }

    WaitForSingleObject(h_tcp, 3000);
    WaitForSingleObject(h_pipe, 3000);

    mem_shutdown();
    release_instance_lock();
    printf("done.\n");
    return 0;
}
