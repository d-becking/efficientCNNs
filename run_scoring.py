import model_scoring
import time

if __name__ == '__main__':
    t1_start = time.perf_counter()
    model_scoring.main()
    t1_stop = time.perf_counter()
    print("Elapsed time: %.2f [s]" % (t1_stop-t1_start))