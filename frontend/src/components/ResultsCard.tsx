
import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Eye } from 'lucide-react';
import { Button } from "@/components/ui/button";

export interface FoulResult {
  id: string;
  videoTitle: string;
  actionClass: string;
  severityRating: number;
  classification: 'Yellow Card' | 'Red Card' | 'No Card';
  timestamp: string;
  videoBlob?: Blob;
}

interface ResultsCardProps {
  result: FoulResult;
}

const ResultsCard = ({ result }: ResultsCardProps) => {
  const videoUrl = result.videoBlob ? URL.createObjectURL(result.videoBlob) : null;

  return (
    <Card className="glass-card overflow-hidden">
      {videoUrl && (
        <div className="aspect-video w-full mb-4">
          <video 
            src={videoUrl} 
            controls 
            className="w-full h-full object-cover"
            onLoadedData={() => {
              return () => URL.revokeObjectURL(videoUrl);
            }}
          />
        </div>
      )}
      <div className="p-4 space-y-3">
        <CardHeader>
          <CardTitle className="text-xl font-semibold text-white truncate">
            {result.videoTitle}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-white/90">Severity</span>
            <Progress 
              value={result.severityRating * 10} 
              className="w-2/3" 
            />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-white/90">Classification</span>
            <span className={`
              px-3 py-1 rounded-full text-sm font-semibold backdrop-blur-sm
              ${result.classification === 'Red Card' 
                ? 'bg-destructive/90 text-white' 
                : result.classification === 'Yellow Card' 
                  ? 'bg-yellow-400/90 text-gray-900' 
                  : 'bg-white/10 text-white'
              }
            `}>
              {result.classification}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-white/90">Type</span>
            <span className="text-sm text-white/90 font-semibold">
              {result.actionClass}
            </span>
          </div>
        </CardContent>
        <CardFooter className="flex justify-between items-center">
          <Button variant="secondary" size="sm" className="glass-effect hover:bg-white/20">
            <Eye className="mr-2 h-4 w-4" /> Interpret
          </Button>
          <span className="text-xs text-white/70">
            {new Date(result.timestamp).toLocaleString()}
          </span>
        </CardFooter>
      </div>
    </Card>
  );
};

export default ResultsCard;
