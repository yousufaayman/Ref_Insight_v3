
import { useState } from 'react';
import Header from '@/components/Header';
import FileUpload from '@/components/FileUpload';
import ResultsSection from '@/components/ResultsSection';
import { useResults } from '@/hooks/useResults';
import { toast } from "sonner";

const Index = () => {
  const { results, processVideos, clearResults } = useResults();
  const [isProcessing, setIsProcessing] = useState(false);

  const handleUploadComplete = async (files: File[]) => {
    setIsProcessing(true);
    try {
      await processVideos(files);
      toast.success(`Successfully processed ${files.length} video${files.length > 1 ? 's' : ''}`);
    } catch (error) {
      toast.error('Failed to process videos. Please try again.');
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    clearResults();
  };

  return (
    <div className="min-h-screen flex flex-col bg-background bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background">
      <Header />
      <main className="flex-1 container max-w-6xl py-8 px-4 sm:px-6 space-y-8">
        <div className="text-center space-y-4 mb-8 glass-effect p-8 rounded-lg">
          <h2 className="text-4xl font-bold tracking-tight text-white">
            Football Foul Detection
          </h2>
          <p className="text-white/80 max-w-2xl mx-auto">
            Upload football game footage for automated foul detection and analysis using machine learning.
          </p>
        </div>
        
        {isProcessing ? (
          <div className="glass-effect rounded-lg flex flex-col items-center justify-center py-16">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary"></div>
            <p className="mt-4 text-lg text-white">Processing videos...</p>
            <p className="text-sm text-white/70 mt-2">This may take a few moments</p>
          </div>
        ) : results.length > 0 ? (
          <ResultsSection results={results} onReset={handleReset} />
        ) : (
          <div className="max-w-xl mx-auto glass-effect p-6 rounded-lg">
            <FileUpload onUploadComplete={handleUploadComplete} />
          </div>
        )}
      </main>
      <footer className="py-6 border-t border-primary/20 glass-effect mt-8">
        <div className="container text-center text-sm text-white/70">
          &copy; {new Date().getFullYear()} REF_INISHGT | AI-Powered Football Analysis
        </div>
      </footer>
    </div>
  );
};

export default Index;
