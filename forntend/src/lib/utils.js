/**
 * Extract the file stem (filename without extension) from any path.
 * Handles Windows backslashes and Unix forward slashes.
 * Examples:
 *   "uploads\video.mp4"  → "video"
 *   "video.mp4"          → "video"
 *   "video"              → "video"   (already a stem)
 */
export function toStem(pathOrStem) {
  if (!pathOrStem) return '';
  // Replace backslashes with forward slashes, then get last segment
  const parts = pathOrStem.replace(/\\/g, '/').split('/');
  let filename = parts[parts.length - 1];
  // Strip (db) flag first — DB stems may contain dots (e.g. from URLs)
  filename = filename.replace(/ \(db\)$/, '');
  // Only strip known video/audio file extensions, not arbitrary dots
  return filename.replace(/\.(mp4|avi|mkv|mov|webm|flv|wmv|mp3|wav|m4a|ogg)$/i, '');
}

/**
 * Return a short display label for a long stem (truncate middle).
 */
export function shortStem(stem, maxLen = 40) {
  if (!stem || stem.length <= maxLen) return stem;
  const half = Math.floor(maxLen / 2);
  return stem.slice(0, half) + '…' + stem.slice(-half);
}
