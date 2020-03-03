import frozen_assignment_retraining
import time

if __name__ == '__main__':
    t1_start = time.perf_counter()
    frozen_assignment_retraining.main()
    t1_stop = time.perf_counter()
    print("Elapsed time: %.2f [s]" % (t1_stop-t1_start))
