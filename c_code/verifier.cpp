#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include "ewn.h"

void Redir_Stdin(const char *testcase) {
    int fd;
    if ((fd = open(testcase, O_RDONLY)) == -1) {
        fprintf(stderr, "open failed: %s: %s\n", strerror(errno), testcase);
        exit(1);
    }
    if (dup2(fd, 0) == -1) {
        perror("dup2 failed");
        exit(1);
    }
    close(fd);
}

int Exec_EWN_Program(const char *program) {
    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) {
        perror("pipe failed");
        exit(1);
    }

    int pid = fork();
    if (pid == -1) {
        perror("fork failed");
        exit(1);
    } 
    else if (pid == 0) {
        lseek(0, 0, SEEK_SET);
        if (dup2(pipe_fd[1], 1) == -1) {
            perror("dup2 failed");
            exit(1);
        }
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        execl(program, program, NULL);
        fprintf(stderr, "exec failed: %s: %s\n", strerror(errno), program);
        exit(1);
    }
    else {
        if (dup2(pipe_fd[0], 0) == -1) {
            perror("dup2 failed");
            exit(1);
        }
        close(pipe_fd[0]);
        close(pipe_fd[1]);
    }

    return pid;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "usage: ./verifier program testcase\n");
        exit(1);
    }

    EWN game;
    int moves[MAX_MOVES];
    int n_move;
    int n_ply;

    Redir_Stdin(argv[2]);
    game.scan_board();
    Exec_EWN_Program(argv[1]);
    wait(NULL);

    printf("ply: 0\n");
    game.print_board();
    printf("\n");

    scanf(" %d", &n_ply);
    for (int i = 0; i < n_ply; i++) {
        int piece, direction;
        scanf(" %d %d", &piece, &direction);
        int move = piece << 4 | direction;
        bool legal = false;

        printf("ply: %d\n", i + 1);
        printf("piece: %d, dir: %d\n", piece, direction);

        n_move = game.move_gen_all(moves);
        for (int j = 0; j < n_move; j++) {
            if (moves[j] == move) {
                legal = true;
                break;
            }
        }
        if (!legal) {
            printf("ILLEGAL!!\n");
            exit(1);
        }

        game.do_move(move);
        game.print_board();
        printf("\n");
    }

    printf("Legal\n");
    return 0;
}
