/*
 * VEC C++ Client SDK
 *
 * Usage:
 *   VecClient vec("localhost", 1920);
 *   int idx = vec.push(vector, 1024);
 *   int idx = vec.push("docs/file.pdf?page=2", vector, 1024);
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
    char label[512];
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

    void fmt_vec(char *cmd, const char *prefix, const float *vec, int dim) {
        char *p = cmd;
        p += sprintf(p, "%s ", prefix);
        for (int i = 0; i < dim; i++) {
            if (i > 0) *p++ = ',';
            p += sprintf(p, "%.6f", vec[i]);
        }
        *p++ = '\n'; *p = '\0';
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

    int push(const float *vec, int dim) {
        char cmd[65536];
        fmt_vec(cmd, "push", vec, dim);
        send_all(cmd, (int)strlen(cmd));
        readline(rbuf, sizeof(rbuf));
        if (strncmp(rbuf, "err", 3) == 0) return -1;
        return atoi(rbuf);
    }

    int push(const char *label, const float *vec, int dim) {
        char cmd[65536];
        char *p = cmd;
        p += sprintf(p, "push %s ", label);
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

    int bpush(const float *vec, int dim) {
        send_all("bpush\n", 6);
        send_all((const char *)vec, dim * (int)sizeof(float));
        readline(rbuf, sizeof(rbuf));
        if (strncmp(rbuf, "err", 3) == 0) return -1;
        return atoi(rbuf);
    }

    int bpush(const char *label, const float *vec, int dim) {
        char header[1024];
        int hlen = sprintf(header, "bpush %s\n", label);
        send_all(header, hlen);
        send_all((const char *)vec, dim * (int)sizeof(float));
        readline(rbuf, sizeof(rbuf));
        if (strncmp(rbuf, "err", 3) == 0) return -1;
        return atoi(rbuf);
    }

    int pull(const float *vec, int dim, VecResult *results, int max_results) {
        char cmd[65536];
        fmt_vec(cmd, "pull", vec, dim);
        send_all(cmd, (int)strlen(cmd));
        readline(rbuf, sizeof(rbuf));
        return parse_results(rbuf, results, max_results);
    }

    int cpull(const float *vec, int dim, VecResult *results, int max_results) {
        char cmd[65536];
        fmt_vec(cmd, "cpull", vec, dim);
        send_all(cmd, (int)strlen(cmd));
        readline(rbuf, sizeof(rbuf));
        return parse_results(rbuf, results, max_results);
    }

    int bpull(const float *vec, int dim, VecResult *results, int max_results) {
        send_all("bpull\n", 6);
        send_all((const char *)vec, dim * (int)sizeof(float));
        readline(rbuf, sizeof(rbuf));
        return parse_results(rbuf, results, max_results);
    }

    int bcpull(const float *vec, int dim, VecResult *results, int max_results) {
        send_all("bcpull\n", 7);
        send_all((const char *)vec, dim * (int)sizeof(float));
        readline(rbuf, sizeof(rbuf));
        return parse_results(rbuf, results, max_results);
    }

    int setLabel(int index, const char *label) {
        char cmd[1024];
        int len = sprintf(cmd, "label %d %s\n", index, label);
        send_all(cmd, len);
        readline(rbuf, sizeof(rbuf));
        return (strncmp(rbuf, "err", 3) == 0) ? -1 : 0;
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
            /* find the last colon in this result (before comma or end) */
            const char *end = strchr(p, ',');
            if (!end) end = p + strlen(p);
            const char *last_colon = NULL;
            for (const char *s = end - 1; s >= p; s--) {
                if (*s == ':') { last_colon = s; break; }
            }
            if (!last_colon) break;

            /* everything before last colon = label or index */
            int label_len = (int)(last_colon - p);
            results[n].distance = (float)atof(last_colon + 1);
            results[n].label[0] = '\0';

            /* check if it's a numeric index or a label */
            char tmp[512];
            if (label_len < (int)sizeof(tmp)) {
                memcpy(tmp, p, label_len);
                tmp[label_len] = '\0';
                char *endptr;
                long idx = strtol(tmp, &endptr, 10);
                if (*endptr == '\0') {
                    results[n].index = (int)idx;
                } else {
                    results[n].index = -1;
                    strncpy(results[n].label, tmp, sizeof(results[n].label) - 1);
                }
            }
            n++;
            p = (*end == ',') ? end + 1 : end;
        }
        return n;
    }
};

#endif /* VEC_CLIENT_H */
