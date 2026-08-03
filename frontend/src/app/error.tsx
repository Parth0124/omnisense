'use client';

export default function Error({ reset }: { error: Error; reset: () => void }) {
  return (
    <div className="p-8">
      <h2>Something went wrong.</h2>
      <button onClick={() => reset()}>Try again</button>
    </div>
  );
}
