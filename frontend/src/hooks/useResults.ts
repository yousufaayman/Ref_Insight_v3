
import { useState } from 'react';
import { FoulResult } from '@/components/ResultsCard';

export const useResults = () => {
  const [results, setResults] = useState<FoulResult[]>([]);

  const processVideos = async (files: File[]) => {
    const mockResults: FoulResult[] = files.map((file, index) => {
      const severityOptions = [3, 5, 7, 8, 9];
      const actionOptions = ['Tackle', 'Pushing', 'Holding', 'Elbowing', 'Sliding'];
      const classificationOptions: ('Yellow Card' | 'Red Card' | 'No Card')[] = ['Yellow Card', 'Red Card', 'No Card'];
      
      const severity = severityOptions[Math.floor(Math.random() * severityOptions.length)];
      let classification: 'Yellow Card' | 'Red Card' | 'No Card';
      
      if (severity <= 4) classification = 'No Card';
      else if (severity <= 7) classification = 'Yellow Card';
      else classification = 'Red Card';
      
      return {
        id: `result-${Date.now()}-${index}`,
        videoTitle: file.name,
        actionClass: actionOptions[Math.floor(Math.random() * actionOptions.length)],
        severityRating: severity,
        classification,
        timestamp: new Date().toISOString(),
        videoBlob: file // Store the video file as a blob
      };
    });
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    setResults(mockResults);
    return mockResults;
  };
  
  const clearResults = () => {
    results.forEach(result => {
      if (result.videoBlob) {
        URL.revokeObjectURL(URL.createObjectURL(result.videoBlob));
      }
    });
    setResults([]);
  };

  return {
    results,
    processVideos,
    clearResults,
  };
};
