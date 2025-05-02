import React, { useState } from 'react';
import { OccurrenceResult } from '@/hooks/useResults';
import { ResultsCard } from './ResultsCard';

interface ResultsSectionProps {
  results: OccurrenceResult[];
  onReset: () => void;
}

export const ResultsSection = ({ results, onReset }: ResultsSectionProps) => {
  const [sortBy, setSortBy] = useState<'severity' | 'time'>('time');

  const sortedResults = [...results].sort((a, b) => {
    if (sortBy === 'severity') {
      return b.severityRating - a.severityRating;
    }
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
        <div className="flex gap-4">
          <button
            onClick={onReset}
            className="px-4 py-2 rounded-md bg-red-600 text-white hover:bg-red-700"
          >
            Reset
          </button>
          <button
            onClick={() => setSortBy('time')}
            className={`px-4 py-2 rounded-md ${
              sortBy === 'time'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Latest First
          </button>
          <button
            onClick={() => setSortBy('severity')}
            className={`px-4 py-2 rounded-md ${
              sortBy === 'severity'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Highest Severity
          </button>
        </div>
      </div>

      <div className="grid gap-6">
        {sortedResults.map((result) => (
          <ResultsCard key={result.id} result={result} />
        ))}
      </div>
    </div>
  );
};
