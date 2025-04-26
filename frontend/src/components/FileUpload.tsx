
import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileVideo } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { toast } from "sonner";

interface FileUploadProps {
  onUploadComplete: (files: File[]) => void;
}

const FileUpload = ({ onUploadComplete }: FileUploadProps) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const videoFiles = acceptedFiles.filter(file => 
      file.type.startsWith('video/')
    );
    
    if (videoFiles.length !== acceptedFiles.length) {
      toast.error('Only video files are accepted');
      return;
    }

    setFiles(prev => [...prev, ...videoFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': []
    },
    multiple: true
  });

  const removeFile = (indexToRemove: number) => {
    setFiles(files.filter((_, index) => index !== indexToRemove));
  };

  const handleSubmit = async () => {
    if (files.length === 0) {
      toast.error('Please select at least one video file');
      return;
    }

    setUploading(true);
    const interval = setInterval(() => {
      setProgress(prev => Math.min(prev + 5, 95));
    }, 200);

    try {
      await onUploadComplete(files);
      setProgress(100);
      toast.success('Upload completed successfully!');
    } catch (error) {
      toast.error('Upload failed. Please try again.');
    } finally {
      clearInterval(interval);
      setUploading(false);
      setFiles([]);
      setProgress(0);
    }
  };

  return (
    <div className="w-full space-y-6">
      <div 
        {...getRootProps()} 
        className={`glass-effect rounded-lg p-8 text-center transition-colors cursor-pointer
          ${isDragActive ? 'bg-primary/20' : 'hover:bg-white/10'}`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-primary mb-4" />
        <p className="text-lg font-medium text-white">Drop video files here</p>
        <p className="text-sm text-white/70 mt-1">Or click to select files</p>
        <p className="text-xs text-white/50 mt-2">Supports all video formats</p>
      </div>
      
      {files.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-white">Selected Files</h3>
          <ul className="space-y-2">
            {files.map((file, index) => (
              <li key={index} className="glass-effect rounded-md p-2 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileVideo className="h-5 w-5 text-primary" />
                  <span className="text-sm text-white truncate max-w-[200px] sm:max-w-[300px]">
                    {file.name}
                  </span>
                  <span className="text-xs text-white/70">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
                <Button 
                  variant="ghost" 
                  size="icon"
                  onClick={() => removeFile(index)}
                  disabled={uploading}
                  className="hover:bg-white/10"
                >
                  <X className="h-4 w-4" />
                  <span className="sr-only">Remove file</span>
                </Button>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {uploading && (
        <div className="space-y-2">
          <Progress value={progress} className="w-full" />
          <p className="text-sm text-right text-white/70">{progress}%</p>
        </div>
      )}
      
      <Button 
        onClick={handleSubmit} 
        disabled={files.length === 0 || uploading}
        className="w-full bg-primary hover:bg-primary/90 text-white"
      >
        {uploading ? 'Processing...' : `Upload and Analyze ${files.length} File${files.length !== 1 ? 's' : ''}`}
      </Button>
    </div>
  );
};

export default FileUpload;
