"""Small fail-fast helpers for bounded Teacher API concurrency.

Only ``max_workers`` jobs are submitted at once.  This prevents an exception
from leaving thousands of queued API jobs running while the executor waits to
exit, and keeps retrying a failed stage cheap because Teacher responses are
already cached separately.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar


Job = TypeVar("Job")
Result = TypeVar("Result")


class BoundedJobError(RuntimeError, Generic[Job]):
    """Wrap a worker exception while retaining the job that produced it."""

    def __init__(self, job: Job, error: BaseException):
        super().__init__(str(error))
        self.job = job
        self.error = error


def run_bounded(
    jobs: Iterable[Job],
    worker: Callable[[Job], Result],
    on_result: Callable[[Job, Result, int], None],
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> int:
    """Run jobs with a bounded queue and stop submitting after first failure."""

    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    iterator = iter(jobs)
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
    )
    futures: dict[Future[Result], Job] = {}
    completed = 0
    failed = True

    def submit_next() -> bool:
        try:
            job = next(iterator)
        except StopIteration:
            return False
        futures[executor.submit(worker, job)] = job
        return True

    try:
        for _ in range(max_workers):
            if not submit_next():
                break

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    raise BoundedJobError(job, error) from error
                completed += 1
                on_result(job, result, completed)
                submit_next()
        failed = False
        return completed
    finally:
        if failed:
            for future in futures:
                future.cancel()
        # On failure, do not wait for queued work. At most max_workers calls are
        # in flight; their own HTTP timeouts bound process shutdown.
        executor.shutdown(wait=not failed, cancel_futures=failed)
