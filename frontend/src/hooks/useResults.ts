import { useState } from 'react';

export interface OccurrenceResult {
  id: string;
  videos: {
    path: string;
    title: string;
  }[];
  actionClass: string;
  severityRating: number;
  classification: 'Yellow Card' | 'Red Card' | 'No Card';
  timestamp: string;
  perViewResults: {
    videoPath: string;
    foulProbabilities: number[];
    offenseProbabilities: number[];
    attentionScore: number[];
  }[];
  aggregatedPredictions: {
    foulProbabilities: number[];
    offenseProbabilities: number[];
    attentionWeights: number[];
  };
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

export const useResults = () => {
  const [results, setResults] = useState<OccurrenceResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processVideos = async (files: File[]) => {
    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('videos', file);
      });

      const response = await fetch(`${API_BASE_URL}/process`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process videos');
      }

      const result: OccurrenceResult = await response.json();
      setResults(prev => [result, ...prev]);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred while processing videos';
      setError(errorMessage);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  };

  const visualizeVideo = async (videoPath: string, attentionScores: number[]) => {
    try {
      const response = await fetch(`${API_BASE_URL}/visualize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          videoPath,
          attentionScores,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to visualize video');
      }

      const result = await response.json();
      return result.visualizationPaths;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred while visualizing video';
      setError(errorMessage);
      throw err;
    }
  };
  
  const clearResults = () => {
    results.forEach(result => {
      result.videos.forEach(video => {
        if (video.path.startsWith('blob:')) {
          URL.revokeObjectURL(video.path);
        }
      });
    });
    setResults([]);
  };

  return {
    results,
    isProcessing,
    error,
    processVideos,
    visualizeVideo,
    clearResults,
  };
};
