import { useState } from 'react';
import { OccurrenceResult } from '@/hooks/useResults';
import { ResultsCard } from './ResultsCard';

interface ResultsSectionProps {
  results: OccurrenceResult[];
}

export const ResultsSection = ({ results }: ResultsSectionProps) => {
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
        <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
        <div className="flex gap-4">
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
          <div
            key={result.id}
            className="bg-white rounded-lg shadow-lg overflow-hidden"
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">
                    {result.actionClass}
                  </h3>
                  <p className="text-sm text-gray-500">
                    {new Date(result.timestamp).toLocaleString()}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-semibold">
                    Severity: {result.severityRating}
                  </span>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      result.classification === 'Red Card'
                        ? 'bg-red-100 text-red-800'
                        : result.classification === 'Yellow Card'
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }`}
                  >
                    {result.classification}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                {result.videos.map((video, index) => (
                  <div
                    key={`${result.id}-${index}`}
                    className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden"
                  >
                    <video
                      src={URL.createObjectURL(video.file)}
                      controls
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 p-2">
                      <p className="text-white text-sm truncate">
                        {video.title}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="border-t pt-4">
                <h4 className="text-lg font-semibold mb-2">Aggregated Analysis</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h5 className="font-medium mb-2">Foul Probabilities</h5>
                    <div className="space-y-1">
                      {result.aggregatedPredictions.foulProbabilities.map(
                        (prob, idx) => (
                          <div
                            key={idx}
                            className="flex items-center gap-2"
                          >
                            <div className="flex-1 h-2 bg-gray-200 rounded-full">
                              <div
                                className="h-full bg-blue-600 rounded-full"
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-600">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h5 className="font-medium mb-2">Offense Probabilities</h5>
                    <div className="space-y-1">
                      {result.aggregatedPredictions.offenseProbabilities.map(
                        (prob, idx) => (
                          <div
                            key={idx}
                            className="flex items-center gap-2"
                          >
                            <div className="flex-1 h-2 bg-gray-200 rounded-full">
                              <div
                                className="h-full bg-red-600 rounded-full"
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-600">
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h5 className="font-medium mb-2">Attention Weights</h5>
                    <div className="space-y-1">
                      {result.aggregatedPredictions.attentionWeights.map(
                        (weight, idx) => (
                          <div
                            key={idx}
                            className="flex items-center gap-2"
                          >
                            <div className="flex-1 h-2 bg-gray-200 rounded-full">
                              <div
                                className="h-full bg-green-600 rounded-full"
                                style={{ width: `${weight * 100}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-600">
                              {(weight * 100).toFixed(1)}%
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
