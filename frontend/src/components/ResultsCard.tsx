import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Eye, X, Loader2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { OccurrenceResult } from '@/hooks/useResults';
import { useResults } from '@/hooks/useResults';
import { toast } from 'sonner';

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
  result: OccurrenceResult;
}

export const ResultsCard = ({ result }: ResultsCardProps) => {
  const [isVisualizing, setIsVisualizing] = useState(false);
  const [visualizationPaths, setVisualizationPaths] = useState<string[] | null>(null);
  const [failedViews, setFailedViews] = useState<Set<number>>(new Set());
  const { visualizeVideo } = useResults();

  const handleInterpret = async () => {
    setIsVisualizing(true);
    setFailedViews(new Set());
    try {
      const visualizationPromises = result.perViewResults.map((viewResult, index) => 
        visualizeVideo(result.videos[index].path, viewResult.attentionScore)
      );
      
      const allPaths = await Promise.all(visualizationPromises);
      const flattenedPaths = allPaths.flat();
      setVisualizationPaths(flattenedPaths);
    } catch (error) {
      toast.error('Failed to generate visualizations');
    } finally {
      setIsVisualizing(false);
    }
  };

  const handleImageError = (viewIndex: number) => {
    setFailedViews(prev => new Set([...prev, viewIndex]));
    toast.error(`Failed to load visualization for View ${viewIndex + 1}`);
  };

  return (
    <Card className="w-full bg-white shadow-xl rounded-xl overflow-hidden">
      <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-2xl font-bold text-gray-900">
              {result.actionClass}
            </CardTitle>
            <p className="text-sm text-gray-600 mt-1">
              {new Date(result.timestamp).toLocaleString()}
            </p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-3">
              <span className="text-lg font-semibold text-gray-900">
                Severity: {result.severityRating.toFixed(2)}
              </span>
              <span
                className={`px-4 py-1.5 rounded-full text-sm font-semibold ${
                  result.classification === 'Red Card'
                    ? 'bg-red-100 text-red-800 border border-red-200'
                    : result.classification === 'Yellow Card'
                    ? 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                    : 'bg-green-100 text-green-800 border border-green-200'
                }`}
              >
                {result.classification}
              </span>
            </div>
            <Button
              onClick={handleInterpret}
              disabled={isVisualizing}
              className={`${
                isVisualizing 
                  ? 'bg-blue-400'
                  : 'bg-blue-600 hover:bg-blue-700'
              } text-white transition-all duration-200 flex items-center gap-2`}
            >
              {isVisualizing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4" />
                  Interpret Results
                </>
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {result.videos.map((video, index) => (
            <div
              key={`${result.id}-${index}`}
              className="relative aspect-video bg-gray-900 rounded-xl overflow-hidden shadow-lg border border-gray-200"
            >
              <video
                src={video.path}
                controls
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-3">
                <p className="text-white text-sm font-medium truncate">
                  View {index + 1}: {video.title}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-none shadow-md">
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-blue-900">Foul Probabilities</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {result.aggregatedPredictions.foulProbabilities.map(
                (prob, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-blue-800 font-medium">
                        {[
                          "No Foul",
                          "Pushing",
                          "Holding",
                          "Tripping",
                          "Kicking",
                          "Striking",
                          "Handball",
                          "Dangerous Play"
                        ][idx]}
                      </span>
                      <span className="text-blue-900 font-semibold">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress 
                      value={prob * 100} 
                      className="h-2 bg-blue-200"
                      indicatorClassName="bg-blue-600"
                    />
                  </div>
                )
              )}
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-red-50 to-red-100 border-none shadow-md">
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-red-900">Offense Probabilities</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {result.aggregatedPredictions.offenseProbabilities.map(
                (prob, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-red-800 font-medium">
                        {[
                          "No Offense",
                          "Offense, without Card",
                          "Serious Offense and Yellow Card",
                          "Violent Offense and Red Card"
                        ][idx]}
                      </span>
                      <span className="text-red-900 font-semibold">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress 
                      value={prob * 100} 
                      className="h-2 bg-red-200"
                      indicatorClassName="bg-red-600"
                    />
                  </div>
                )
              )}
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-50 to-green-100 border-none shadow-md">
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-green-900">Attention Weights</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {result.aggregatedPredictions.attentionWeights.map(
                (weight, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-green-800 font-medium">
                        Frame Group {idx + 1}
                      </span>
                      <span className="text-green-900 font-semibold">
                        {(weight * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress 
                      value={weight * 100} 
                      className="h-2 bg-green-200"
                      indicatorClassName="bg-green-600"
                    />
                  </div>
                )
              )}
            </CardContent>
          </Card>
        </div>
      </CardContent>

      {/* Visualization Modal */}
      {visualizationPaths && (
        <div className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
          <div className="bg-gray-900 rounded-xl p-6 max-w-[95vw] w-full max-h-[95vh] overflow-y-auto border border-gray-700">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-2xl font-bold text-white">Attention Rollout Visualization</h3>
              <Button
                onClick={() => setVisualizationPaths(null)}
                variant="outline"
                size="icon"
                className="hover:bg-gray-800 border-gray-700 text-gray-300"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {result.videos.map((_, viewIndex) => (
                <div key={viewIndex} className="relative bg-black rounded-xl overflow-hidden shadow-2xl border border-gray-800">
                  <div className="absolute top-3 left-3 z-10">
                    <span className="px-3 py-1.5 bg-black/70 text-white text-sm font-medium rounded-full border border-gray-700">
                      View {viewIndex + 1}
                    </span>
                  </div>
                  {viewIndex < (visualizationPaths?.length || 0) && !failedViews.has(viewIndex) ? (
                    <div className="aspect-video relative group">
                      <img
                        src={`${import.meta.env.VITE_API_BASE_URL}/visualization/${visualizationPaths[viewIndex]}`}
                        alt={`Attention Rollout Visualization - View ${viewIndex + 1}`}
                        className="w-full h-full object-contain bg-black"
                        style={{
                          imageRendering: 'crisp-edges',
                          maxHeight: '80vh'
                        }}
                        onError={() => handleImageError(viewIndex)}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    </div>
                  ) : (
                    <div className="aspect-video flex items-center justify-center bg-gray-900">
                      <p className="text-gray-400 text-center p-4">
                        {failedViews.has(viewIndex) 
                          ? `Failed to load visualization for View ${viewIndex + 1}`
                          : 'No visualization available'
                        }
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-6 text-gray-400 text-sm">
              <p>* Brighter areas indicate higher attention weights in the model's decision-making process</p>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default ResultsCard;
