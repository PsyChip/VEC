/*
 * VEC C++ Client SDK
 *
 * Usage:
 *   VecClient vec("localhost", 1920);
 *   int idx = vec.push(vector, 1024);
 *   VecResult results[10];
 *   int n = vec.pull(vector, 1024, results, 10);
 *   vec.close();
 *
 * Build: link with ws2_32.lib (Windows) or nothing extra (Linux)
 */
#ifndef VEC_CLIENT_H
#define VEC_CLIENT_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET vec_sock_t;
#define VEC_INVALID_SOCK INVALID_SOCKET
#define vec_close_sock closesocket
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
typedef int vec_sock_t;
#define VEC_INVALID_SOCK -1
#define vec_close_sock close
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct VecResult {
    int index;
    float distance;
};

class VecClient {
    vec_sock_t sock;
    char rbuf[65536];

    int readline(char *buf, int maxlen) {
        int total = 0;
        while (total < maxlen - 1) {
            int r = recv(sock, buf + total, 1, 0);
            if (r <= 0) break;
            total++;
            if (buf[total - 1] == '\n') break;
        }
        buf[total] = '\0';
        if (total > 0 && buf[total - 1] == '\n') buf[--total] = '\0';
        return total;
    }

    int send_all(const char *data, int len) {
        int sent = 0;
        while (sent < len) {
            int r = send(sock, data + sent, len - sent, 0);
            if (r <= 0) return -1;
            sent += r;
        }
        return sent;
    }

public:
    VecClient() : sock(VEC_INVALID_SOCK) {}

    int connect_tcp(const char *host, int port) {
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == VEC_INVALID_SOCK) return -1;

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons((unsigned short)port);

        if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
            struct hostent *he = gethostbyname(host);
            if (!he) { vec_close_sock(sock); sock = VEC_INVALID_SOCK; return -1; }
            memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
        }

        if (::connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
            vec_close_sock(sock); sock = VEC_INVALID_SOCK; return -1;
        }
        return 0;
    }

#ifndef _WIN32
    int connect_unix(const char *name) {
        char path[256];
        snprintf(path, sizeof(path), "/tmp/vec_%s.sock", name);
        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock < 0) return -1;
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
        if (::connect(sock, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
            ::close(sock); sock = VEC_INVALID_SOCK; return -1;
        }
        return 0;
    }
#endif

#ifdef _WIN32
    int connect_pipe(const char *name) {
        char pipe_name[256];
        snprintf(pipe_name, sizeof(pipe_name), "\\\\.\\pipe\\vec_%s", name);
        HANDLE pipe = CreateFileA(pipe_name, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
        if (pipe == INVALID_HANDLE_VALUE) return -1;
        /* wrap pipe handle as socket - not directly compatible, use TCP instead */
        CloseHandle(pipe);
        return -1; /* pipe mode not supported in this SDK, use TCP */
    }
#endif

    int push(const float *vec, int dim) {
        char cmd[65536];
        char *p = cmd;
        p += sprintf(p, "push ");
        for (int i = 0; i < dim; i++) {
            if (i > 0) *p++ = ',';
            p += sprintf(p, "%.6f", vec[i]);
        }
        *p++ = '\n'; *p = '\0';
        send_all(cmd, (int)(p - cmd));
        readline(rbuf, sizeof(rbuf));
        if (strncmp(rbuf, "err", 3) == 0) return -1;
        return atoi(rbuf);
    }

    int pull(const float *vec, int dim, VecResult *results, int max_results) {
        char cmd[65536];
        char *p = cmd;
        p += sprintf(p, "pull ");
        for (int i = 0; i < dim; i++) {
            if (i > 0) *p++ = ',';
            p += sprintf(p, "%.6f", vec[i]);
        }
        *p++ = '\n'; *p = '\0';
        send_all(cmd, (int)(p - cmd));
        readline(rbuf, sizeof(rbuf));
        return parse_results(rbuf, results, max_results);
    }

    int cpull(const float *vec, int dim, VecResult *results, int max_results) {
        char cmd[65536];
        char *p = cmd;
        p += sprintf(p, "cpull ");
        for (int i = 0; i < dim; i++) {
            if (i > 0) *p++ = ',';
            p += sprintf(p, "%.6f", vec[i]);
        }
        *p++ = '\n'; *p = '\0';
        send_all(cmd, (int)(p - cmd));
        readline(rbuf, sizeof(rbuf));
        return parse_results(rbuf, results, max_results);
    }

    int bpush(const float *vectors, int count, int dim) {
        char header[64];
        int hlen = sprintf(header, "bpush %d\n", count);
        send_all(header, hlen);
        send_all((const char *)vectors, count * dim * (int)sizeof(float));
        readline(rbuf, sizeof(rbuf));
        if (strncmp(rbuf, "err", 3) == 0) return -1;
        return atoi(rbuf);
    }

    int vec_delete(int index) {
        char cmd[64];
        int len = sprintf(cmd, "delete %d\n", index);
        send_all(cmd, len);
        readline(rbuf, sizeof(rbuf));
        return (strncmp(rbuf, "err", 3) == 0) ? -1 : 0;
    }

    int undo() {
        send_all("undo\n", 5);
        readline(rbuf, sizeof(rbuf));
        return (strncmp(rbuf, "err", 3) == 0) ? -1 : 0;
    }

    int save() {
        send_all("save\n", 5);
        readline(rbuf, sizeof(rbuf));
        return (strncmp(rbuf, "err", 3) == 0) ? -1 : 0;
    }

    int size() {
        send_all("size\n", 5);
        readline(rbuf, sizeof(rbuf));
        return atoi(rbuf);
    }

    void close() {
        if (sock != VEC_INVALID_SOCK) {
            vec_close_sock(sock);
            sock = VEC_INVALID_SOCK;
        }
    }

    int parse_results(const char *resp, VecResult *results, int max_results) {
        if (strncmp(resp, "err", 3) == 0) return -1;
        int n = 0;
        const char *p = resp;
        while (*p && n < max_results) {
            results[n].index = atoi(p);
            const char *colon = strchr(p, ':');
            if (!colon) break;
            results[n].distance = (float)atof(colon + 1);
            n++;
            const char *comma = strchr(p, ',');
            if (!comma) break;
            p = comma + 1;
        }
        return n;
    }
};

#endif /* VEC_CLIENT_H */
