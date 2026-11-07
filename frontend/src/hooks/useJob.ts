// useJob hook — polls /status until the job reaches a terminal state

import { useState, useEffect, useCallback, useRef } from 'react';
import { getJobStatus } from '../api/client';
import type { JobResponse, JobStatus } from '../types';

const TERMINAL: JobStatus[] = ['completed', 'completed_without_prediction', 'failed', 'cancelled'];
const POLL_INTERVAL_MS = 2000;

interface UseJobState {
  job: JobResponse | null;
  loading: boolean;
  error: string | null;
  stopPolling: () => void;
  startPolling: (jobId: string) => void;
}

export function useJob(initialJobId?: string): UseJobState {
  const [job, setJob] = useState<JobResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const jobIdRef = useRef<string | null>(initialJobId ?? null);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const poll = useCallback(async (jobId: string) => {
    try {
      const data = await getJobStatus(jobId);
      setJob(data);
      setLoading(false);
      if (TERMINAL.includes(data.status)) {
        stopPolling();
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to fetch job status';
      setError(msg);
      setLoading(false);
      stopPolling();
    }
  }, [stopPolling]);

  const startPolling = useCallback((jobId: string) => {
    stopPolling();
    jobIdRef.current = jobId;
    setLoading(true);
    setError(null);
    poll(jobId);
    intervalRef.current = setInterval(() => poll(jobId), POLL_INTERVAL_MS);
  }, [poll, stopPolling]);

  useEffect(() => {
    if (initialJobId) startPolling(initialJobId);
    return stopPolling;
  }, [initialJobId]); // eslint-disable-line react-hooks/exhaustive-deps

  return { job, loading, error, stopPolling, startPolling };
}
