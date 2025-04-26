
import { useState } from 'react';
import ResultsCard, { FoulResult } from './ResultsCard';
import { Button } from "@/components/ui/button";

interface ResultsSectionProps {
  results: FoulResult[];
  onReset: () => void;
}

const ResultsSection = ({ results, onReset }: ResultsSectionProps) => {
  const [sortBy, setSortBy] = useState<'severity' | 'time'>('time');
  
  const sortedResults = [...results].sort((a, b) => {
    if (sortBy === 'severity') {
      return b.severityRating - a.severityRating;
    } else {
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    }
  });

  return (
    <div className="w-full space-y-6">
      <div className="glass-effect p-4 rounded-lg flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
        <div className="flex space-x-2">
          <Button 
            variant={sortBy === 'time' ? 'secondary' : 'ghost'} 
            size="sm"
            onClick={() => setSortBy('time')}
            className={sortBy !== 'time' ? 'hover:bg-white/10 text-white' : ''}
          >
            Latest
          </Button>
          <Button 
            variant={sortBy === 'severity' ? 'secondary' : 'ghost'} 
            size="sm"
            onClick={() => setSortBy('severity')}
            className={sortBy !== 'severity' ? 'hover:bg-white/10 text-white' : ''}
          >
            Severity
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sortedResults.map((result) => (
          <ResultsCard key={result.id} result={result} />
        ))}
      </div>
      
      <div className="flex justify-center pt-4">
        <Button 
          onClick={onReset} 
          variant="outline" 
          className="glass-effect hover:bg-white/10 text-white"
        >
          Analyze New Videos
        </Button>
      </div>
    </div>
  );
};

export default ResultsSection;
