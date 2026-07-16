import { useState, useEffect, useRef, useCallback } from 'react';
import { getStatus } from '@/lib/api';

export function useStatus(intervalMs = 3000, enabled = true) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const timerRef = useRef(null);
  const activeRef = useRef(true);
  const fetchingRef = useRef(false);

  const fetch = useCallback(async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;
    try {
      const data = await getStatus();
      if (activeRef.current) {
        setStatus(data);
        setError(null);
      }
    } catch (e) {
      if (activeRef.current) setError(e.message);
    } finally {
      fetchingRef.current = false;
      if (activeRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      activeRef.current = false;
      fetchingRef.current = false;
      clearInterval(timerRef.current);
      setStatus(null);
      setLoading(false);
      return;
    }
    activeRef.current = true;
    setLoading(true);
    fetch();
    timerRef.current = setInterval(fetch, intervalMs);
    return () => {
      activeRef.current = false;
      clearInterval(timerRef.current);
    };
  }, [fetch, intervalMs, enabled]);

  return { status, loading, error, refetch: fetch };
}
