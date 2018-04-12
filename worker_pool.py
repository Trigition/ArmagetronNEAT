#!/usr/bin/env python
# -*- coding: utf-8 -*-


from queue import Queue
from threading import Thread


class Worker(Thread):

    """Thread independent worker that performs tasks"""

    def __init__(self, tasks, results):
        """Initializes a worker

        :tasks: TODO

        """
        Thread.__init__(self)

        self.tasks = tasks
        self.daemon = True
        self.results = results
        self.start()

    def run(self):
        """Runs the worker
        :returns: TODO

        """
        while True:
            func, args, kargs = self.tasks.get()
            try:
                result = func(*args, **kargs)
                self.results.append(result)
            except Exception as e:
                print(e)
            finally:
                self.tasks.task_done()

class Worker_Pool:

    """A container of threads/workers"""

    def __init__(self, n_threads):
        """Initializes a Pool of Workers

        :n_threads: Number of threads to initialize

        """

        self.tasks = Queue(n_threads)
        self.results = []
        self.workers = []

        for _ in range(n_threads):
            worker = Worker(self.tasks, self.results)
            self.workers.append(worker)

    def reset_results(self):
        self.results = []
        for worker in self.workers:
            worker.results = self.results

    def add_task(self, func, *args, **kargs):
        """Adds a task to the queue

        :func: The target function
        :*args: Arguments for the function
        :**kargs: Additional arguments for the function

        """
        self.tasks.put((func, args, kargs))

    def wait_for_completion(self):
        """Wait for all tasks to finish
        :returns: TODO

        """
        self.tasks.join()
        
