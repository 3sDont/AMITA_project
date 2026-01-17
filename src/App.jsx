import { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import toast, { Toaster } from 'react-hot-toast';

export default function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordingSource, setRecordingSource] = useState('microphone'); // 'microphone' or 'system'
  const [uploadedFilename, setUploadedFilename] = useState(null); // Store backend filename
  const [processingMode, setProcessingMode] = useState('flow'); // 'flash', 'flow', 'deep'
  const [activeSegmentIndex, setActiveSegmentIndex] = useState(null); // Track which segment is currently playing
  const [processingProgress, setProcessingProgress] = useState(0); // 0-100
  const [processingStage, setProcessingStage] = useState(''); // Current stage name
  const [processingStartTime, setProcessingStartTime] = useState(null);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(null);
  const [searchQuery, setSearchQuery] = useState(''); // Search transcript
  const [processingHistory, setProcessingHistory] = useState(() => {
    const saved = localStorage.getItem('amita_history');
    return saved ? JSON.parse(saved) : [];
  });
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const recordingInterval = useRef(null);
  const processingAbortController = useRef(null);
  const progressInterval = useRef(null);

  const [transcript, setTranscript] = useState(() => {
    const saved = localStorage.getItem('amita_transcript');
    return saved ? JSON.parse(saved) : [];
  });
  const [summary, setSummary] = useState(() => {
    return localStorage.getItem('amita_summary') || "";
  });
  const [tasks, setTasks] = useState(() => {
    const saved = localStorage.getItem('amita_tasks');
    return saved ? JSON.parse(saved) : [];
  });

  const API_URL = "http://localhost:8000";

  // Save to localStorage whenever data changes
  useEffect(() => {
    if (transcript.length > 0) {
      localStorage.setItem('amita_transcript', JSON.stringify(transcript));
    }
  }, [transcript]);

  useEffect(() => {
    if (summary) {
      localStorage.setItem('amita_summary', summary);
    }
  }, [summary]);

  useEffect(() => {
    if (tasks.length > 0) {
      localStorage.setItem('amita_tasks', JSON.stringify(tasks));
    }
  }, [tasks]);

  // Save processing history
  useEffect(() => {
    if (processingHistory.length > 0) {
      localStorage.setItem('amita_history', JSON.stringify(processingHistory));
    }
  }, [processingHistory]);

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Clear previous data
    setUploadedFilename(null);
    setTranscript([]);
    setSummary("");
    setTasks([]);
    
    setAudioFile(URL.createObjectURL(file));
    setUploadedFile(file);

    // Upload to backend but DON'T auto-process
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("[UPLOAD] Upload successful:", data);
        console.log("[UPLOAD] Setting uploadedFilename to:", data.filename);
        setUploadedFilename(data.filename); // Store filename for later processing
      }
    } catch (error) {
      console.error("[UPLOAD] Upload failed:", error);
      toast.error("Upload failed. Make sure backend server is running!");
    }
  };

  const handleRunProcessing = async () => {
    if (!uploadedFilename) {
      toast.error("Please upload or record an audio file first!");
      return;
    }
    console.log("Processing file:", uploadedFilename, "with mode:", processingMode);
    await processAudio(uploadedFilename, processingMode);
  };

  // Cancel processing
  const handleCancelProcessing = async () => {
    if (processingAbortController.current) {
      processingAbortController.current.abort();
      console.log("🛑 Processing cancelled by user");
    }
    if (progressInterval.current) {
      clearInterval(progressInterval.current);
    }
    
    // Try to cancel backend processing
    if (uploadedFilename) {
      try {
        await fetch(`${API_URL}/api/cancel/${uploadedFilename}`, {
          method: "POST",
        });
        console.log("✅ Backend cancel request sent");
      } catch (error) {
        console.log("⚠️ Could not send cancel to backend:", error);
      }
    }
    
    setIsProcessing(false);
    setProcessingProgress(0);
    setProcessingStage('');
    setEstimatedTimeRemaining(null);
    setTranscript([]);
    setSummary("");
    setTasks([]);
  };

  // Simulate progress stages - More realistic estimation based on audio duration
  const simulateProgress = (mode, audioDuration = 60) => {
    // Calculate actual estimated time based on mode and audio duration
    const processingSpeed = {
      'flash': 0.5,  // 30s for 1min audio
      'flow': 1.5,   // 90s for 1min audio  
      'deep': 3.0    // 180s for 1min audio
    };
    
    const totalEstimatedSeconds = Math.ceil(audioDuration * processingSpeed[mode]);
    
    const stages = [
      { name: '⬆️ Uploading & Preprocessing', progress: 10, percentage: 0.10 },
      { name: '🎵 Audio Analysis', progress: 25, percentage: 0.15 },
      { name: '🗣️ Speech Recognition', progress: 60, percentage: 0.50 },
      { name: '👥 Speaker Identification', progress: 85, percentage: 0.20 },
      { name: '🤖 AI Analysis & Summary', progress: 95, percentage: 0.05 },
    ];

    let currentStageIndex = 0;
    let currentProgress = 0;
    setProcessingStage(stages[0].name);
    setProcessingProgress(5);
    setEstimatedTimeRemaining(totalEstimatedSeconds);

    const updateInterval = 500; // Update every 500ms
    const progressPerUpdate = 100 / (totalEstimatedSeconds * (1000 / updateInterval));

    progressInterval.current = setInterval(() => {
      currentProgress += progressPerUpdate;
      
      // Don't exceed 98% until actually complete
      if (currentProgress > 98) currentProgress = 98;
      
      // Update stage based on progress
      const currentStage = stages.find((s, idx) => 
        currentProgress < s.progress && (idx === 0 || currentProgress >= stages[idx - 1].progress)
      ) || stages[stages.length - 1];
      
      setProcessingStage(currentStage.name);
      setProcessingProgress(Math.floor(currentProgress));
      
      // Calculate remaining time
      const elapsed = (Date.now() - processingStartTime) / 1000;
      const remaining = Math.max(0, Math.ceil(totalEstimatedSeconds - elapsed));
      setEstimatedTimeRemaining(remaining);
      
    }, updateInterval);
  };

  const processAudio = async (filename, mode = 'flow') => {
    // Create abort controller
    processingAbortController.current = new AbortController();
    
    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingStartTime(Date.now());
    setTranscript([{ time: "00:00", speaker: "System", text: "Processing audio... Please wait..." }]);
    setSummary("Processing...");
    setTasks(["Audio processing in progress..."]);

    // Start progress simulation - use audio duration if available
    const audioDuration = duration || 60; // fallback to 60s if not available
    simulateProgress(mode, audioDuration);

    try {
      const response = await fetch(`${API_URL}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, mode }),
        signal: processingAbortController.current.signal,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("✅ Processing response:", data);
        console.log("📝 Transcript:", data.transcript?.length, "items");
        console.log("📄 Summary:", data.summary?.substring(0, 50));
        console.log("✓ Tasks:", data.tasks?.length, "items");
        
        // Clear progress interval
        if (progressInterval.current) {
          clearInterval(progressInterval.current);
        }
        
        // Set final progress
        setProcessingProgress(100);
        setProcessingStage('✅ Complete');
        setEstimatedTimeRemaining(0);
        
        setTranscript(data.transcript);
        setSummary(data.summary);
        setTasks(data.tasks);
        
        // Save to processing history
        const processingTime = Math.ceil((Date.now() - processingStartTime) / 1000);
        const historyEntry = {
          id: Date.now(),
          filename: filename,
          mode: mode,
          timestamp: new Date().toISOString(),
          processingTime: processingTime,
          segmentCount: data.transcript?.length || 0,
          success: true
        };
        
        setProcessingHistory(prev => [historyEntry, ...prev.slice(0, 9)]); // Keep last 10
        
        console.log(`✅ Processing completed in ${processingTime}s`);
      } else {
        throw new Error("Processing failed");
      }
    } catch (error) {
      // Clear progress interval on error
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
      }
      
      if (error.name === 'AbortError') {
        console.log("Processing was cancelled");
        setTranscript([{ time: "00:00", speaker: "System", text: "Processing cancelled by user." }]);
        setSummary("Processing was cancelled.");
        setTasks([]);
      } else {
        console.error("Processing failed:", error);
        setTranscript([{ time: "00:00", speaker: "Error", text: "Processing failed. Check backend server." }]);
        setSummary("Error occurred during processing.");
        setTasks(["Please try again"]);
      }
    } finally {
      setIsProcessing(false);
      setProcessingProgress(0);
      setProcessingStage('');
      setEstimatedTimeRemaining(null);
    }
  };

  useEffect(() => {
    if (audioFile) {
      if (wavesurfer.current) {
        wavesurfer.current.destroy();
      }

      wavesurfer.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: "#8ecae6",
        progressColor: "#219ebc",
        cursorColor: "#023047",
        height: 80,
      });

      wavesurfer.current.load(audioFile);

      // Update time and duration
      wavesurfer.current.on('ready', () => {
        setDuration(wavesurfer.current.getDuration());
      });

      wavesurfer.current.on('audioprocess', () => {
        setCurrentTime(wavesurfer.current.getCurrentTime());
        updateActiveSegment(wavesurfer.current.getCurrentTime());
      });

      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
        setActiveSegmentIndex(null);
      });
    }
  }, [audioFile]);

  const togglePlay = () => {
    if (wavesurfer.current) {
      wavesurfer.current.playPause();
      setIsPlaying(!isPlaying);
    }
  };

  const skipBackward = () => {
    if (wavesurfer.current) {
      const newTime = Math.max(0, wavesurfer.current.getCurrentTime() - 5);
      wavesurfer.current.seekTo(newTime / wavesurfer.current.getDuration());
    }
  };

  const skipForward = () => {
    if (wavesurfer.current) {
      const duration = wavesurfer.current.getDuration();
      const newTime = Math.min(duration, wavesurfer.current.getCurrentTime() + 5);
      wavesurfer.current.seekTo(newTime / duration);
    }
  };

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Parse timestamp string (MM:SS) to seconds
  const parseTimestamp = (timeString) => {
    const parts = timeString.split(':');
    if (parts.length === 2) {
      const minutes = parseInt(parts[0], 10);
      const seconds = parseInt(parts[1], 10);
      return minutes * 60 + seconds;
    }
    return 0;
  };

  // Handle timestamp click - jump audio to that position
  const handleTimestampClick = (timeString, index) => {
    if (!wavesurfer.current || !audioFile) {
      toast.error('⚠️ Please upload an audio file first!');
      return;
    }

    const targetSeconds = parseTimestamp(timeString);
    const targetPosition = targetSeconds / duration; // Position from 0 to 1
    
    // Jump to position
    wavesurfer.current.seekTo(targetPosition);
    
    // Set active segment
    setActiveSegmentIndex(index);
    
    // Auto-play if not playing
    if (!isPlaying) {
      wavesurfer.current.play();
      setIsPlaying(true);
    }
  };

  // Update active segment based on current audio time
  const updateActiveSegment = (currentSeconds) => {
    if (transcript.length === 0) return;

    // Find which segment matches current time
    for (let i = 0; i < transcript.length; i++) {
      const segmentTime = parseTimestamp(transcript[i].time);
      const nextSegmentTime = i < transcript.length - 1 
        ? parseTimestamp(transcript[i + 1].time) 
        : duration;

      if (currentSeconds >= segmentTime && currentSeconds < nextSegmentTime) {
        if (activeSegmentIndex !== i) {
          setActiveSegmentIndex(i);
        }
        break;
      }
    }
  };

  // Recording functions
  const startRecording = async () => {
    try {
      // Clear previous data
      setUploadedFilename(null);
      setTranscript([]);
      setSummary("");
      setTasks([]);
      
      let stream;
      if (recordingSource === 'system') {
        // Capture system audio using getDisplayMedia
        stream = await navigator.mediaDevices.getDisplayMedia({
          video: {
            displaySurface: "monitor"
          },
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false
          }
        });
        
        // Stop video track if not needed, keep only audio
        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
          videoTrack.stop();
          stream.removeTrack(videoTrack);
        }
      } else {
        // Capture from microphone
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Convert to File object
        const file = new File([audioBlob], `recording_${Date.now()}.webm`, { type: 'audio/webm' });
        
        setAudioFile(audioUrl);
        setUploadedFile(file);
        setRecordingTime(0);
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());

        // Auto-upload recorded audio
        await uploadRecordedAudio(file);
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      
      // Start recording timer
      recordingInterval.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast.error('Could not access microphone. Please grant permission.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
      
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
      }
    }
  };

  const uploadRecordedAudio = async (file) => {
    console.log("[RECORDING] Uploading recorded file:", file.name, "Size:", file.size);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("[RECORDING] Upload successful:", data);
        console.log("[RECORDING] Setting uploadedFilename to:", data.filename);
        setUploadedFilename(data.filename); // Store filename for later processing
      }
    } catch (error) {
      console.error("[RECORDING] Upload failed:", error);
      toast.error("Upload failed. Make sure backend server is running!");
    }
  };

  // Download functions
  const downloadAudio = () => {
    if (!audioFile) return;
    const link = document.createElement('a');
    link.href = audioFile;
    link.download = uploadedFile?.name || `recording_${Date.now()}.webm`;
    link.click();
  };

  const downloadDialog = () => {
    if (transcript.length === 0) return;
    
    let content = "DIALOG / TRANSCRIPT\n";
    content += "=" .repeat(50) + "\n\n";
    
    transcript.forEach((line) => {
      content += `[${line.time}] ${line.speaker}:\n${line.text}\n\n`;
    });
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `dialog_${Date.now()}.txt`;
    link.click();
  };

  const downloadSummary = () => {
    if (!summary) return;
    
    let content = "MEETING SUMMARY\n";
    content += "=" .repeat(50) + "\n\n";
    content += summary;
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `summary_${Date.now()}.txt`;
    link.click();
  };

  const downloadTasks = () => {
    if (tasks.length === 0) return;
    
    let content = "ACTION ITEMS / TASKS\n";
    content += "=" .repeat(50) + "\n\n";
    
    tasks.forEach((task, index) => {
      const taskText = typeof task === 'string' ? task : task.task || JSON.stringify(task);
      content += `${index + 1}. ${taskText}\n`;
      
      if (typeof task === 'object' && task !== null) {
        if (task.assigned_to) content += `   👤 Người đảm nhiệm: ${task.assigned_to}\n`;
        if (task.deadline) content += `   📅 Deadline: ${task.deadline}\n`;
        if (task.priority) content += `   ⚡ Priority: ${task.priority}\n`;
        if (task.how_to) content += `   📝 Cách thực hiện: ${task.how_to}\n`;
      }
      content += "\n";
    });
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `tasks_${Date.now()}.txt`;
    link.click();
  };

  // Format markdown-like text to HTML with bold headers and bullet points
  const formatSummary = (text) => {
    if (!text) return '';
    
    // Split by lines
    const lines = text.split('\n');
    let html = '';
    
    for (let i = 0; i < lines.length; i++) {
      let line = lines[i];
      
      // Check if line starts with emoji icons (🎯, 📋, ✅, etc.)
      const emojiHeaderMatch = line.match(/^([🎯📋✅💡🔍📊🗣️💬🤝📌⚡🎤📝]+)\s*(.+?):\s*$/);
      if (emojiHeaderMatch) {
        const emoji = emojiHeaderMatch[1];
        const headerText = emojiHeaderMatch[2];
        html += `<div class="font-bold text-lg mt-4 mb-2 text-purple-700">${emoji} ${headerText}:</div>`;
        continue;
      }
      
      // Check for **Bold Header**: pattern
      const boldHeaderMatch = line.match(/^\*\*(.+?)\*\*:\s*$/);
      if (boldHeaderMatch) {
        html += `<div class="font-bold text-lg mt-4 mb-2 text-purple-700">${boldHeaderMatch[1]}:</div>`;
        continue;
      }
      
      // Check for bullet points starting with "- "
      if (line.trim().startsWith('- ')) {
        let bulletText = line.trim().substring(2);
        
        // Handle **bold text**: within bullet points
        bulletText = bulletText.replace(/\*\*(.+?)\*\*:/g, '<strong class="text-purple-600">$1:</strong>');
        
        html += `<div class="ml-4 mb-2 flex items-start gap-2">
          <span class="text-purple-500 mt-1">•</span>
          <span class="flex-1">${bulletText}</span>
        </div>`;
        continue;
      }
      
      // Empty lines
      if (line.trim() === '') {
        html += '<br/>';
        continue;
      }
      
      // Regular text
      html += `<div class="mb-1">${line}</div>`;
    }
    
    return html;
  };

  // Filter transcript based on search query
  const filteredTranscript = transcript.filter(line => {
    if (!searchQuery.trim()) return true;
    
    const query = searchQuery.toLowerCase();
    return (
      line.speaker?.toLowerCase().includes(query) ||
      line.text?.toLowerCase().includes(query) ||
      line.time?.includes(query)
    );
  });

  // Highlight matching text in transcript
  const highlightText = (text, query) => {
    if (!query.trim()) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);
    
    return parts.map((part, index) => 
      regex.test(part) 
        ? `<mark class="bg-yellow-300 text-gray-900 px-1 rounded">${part}</mark>`
        : part
    ).join('');
  };

  const downloadAll = () => {
    if (transcript.length === 0 && !summary && tasks.length === 0) return;
    
    let content = "AMITA - MEETING REPORT\n";
    content += "=" .repeat(70) + "\n";
    content += `Generated: ${new Date().toLocaleString()}\n`;
    content += "=" .repeat(70) + "\n\n";
    
    if (transcript.length > 0) {
      content += "DIALOG / TRANSCRIPT\n";
      content += "-" .repeat(70) + "\n";
      transcript.forEach((line) => {
        content += `[${line.time}] ${line.speaker}: ${line.text}\n`;
      });
      content += "\n\n";
    }
    
    if (summary) {
      content += "SUMMARY\n";
      content += "-" .repeat(70) + "\n";
      content += summary + "\n\n\n";
    }
    
    if (tasks.length > 0) {
      content += "ACTION ITEMS / TASKS\n";
      content += "-" .repeat(70) + "\n";
      tasks.forEach((task, index) => {
        const taskText = typeof task === 'string' ? task : task.task || JSON.stringify(task);
        content += `${index + 1}. ${taskText}\n`;
        
        if (typeof task === 'object' && task !== null) {
          if (task.assigned_to) content += `   👤 Người đảm nhiệm: ${task.assigned_to}\n`;
          if (task.deadline) content += `   📅 Deadline: ${task.deadline}\n`;
          if (task.priority) content += `   ⚡ Priority: ${task.priority}\n`;
          if (task.how_to) content += `   📝 Cách thực hiện: ${task.how_to}\n`;
        }
        content += "\n";
      });
    }
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `amita_report_${Date.now()}.txt`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header Section with Modern Gradient */}
      <div className="relative bg-gradient-to-br from-indigo-600 via-blue-600 to-cyan-500 text-white py-20 px-4 overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute top-0 left-0 w-full h-full opacity-10">
          <div className="absolute top-10 left-10 w-72 h-72 bg-white rounded-full blur-3xl"></div>
          <div className="absolute bottom-10 right-10 w-96 h-96 bg-white rounded-full blur-3xl"></div>
        </div>
        
        <div className="max-w-6xl mx-auto text-center relative z-10">
          <div className="inline-flex items-center gap-3 mb-4 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full border border-white/20">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
            <span className="text-sm font-medium">AI-Powered Meeting Assistant</span>
          </div>
          <h1 className="text-6xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white to-blue-100">
            AMITA
          </h1>
          <p className="text-xl text-blue-50 font-light tracking-wide">
            Meeting Insight & Task Assistant
          </p>
          <p className="text-sm text-blue-100 mt-2 max-w-2xl mx-auto">
            Transform your meetings into actionable insights with AI-powered transcription, summarization, and task extraction
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-10">
        {/* Upload and Record Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          {/* Upload Audio */}
          <div className="group bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div>
                <label className="font-bold text-gray-800 text-lg block">
                  Upload Audio File
                </label>
                {isProcessing && (
                  <span className="text-xs text-blue-600 font-medium animate-pulse">
                    Processing...
                  </span>
                )}
              </div>
            </div>
            <div className="relative">
              <input
                type="file"
                accept="audio/*"
                onChange={handleUpload}
                disabled={isProcessing || isRecording}
                className="block w-full text-sm text-gray-600
                  file:mr-4 file:py-3 file:px-6
                  file:rounded-xl file:border-0
                  file:text-sm file:font-bold
                  file:bg-gradient-to-r file:from-blue-600 file:to-indigo-600 file:text-white
                  hover:file:from-blue-700 hover:file:to-indigo-700
                  file:cursor-pointer cursor-pointer
                  file:transition-all file:duration-300
                  file:shadow-md hover:file:shadow-lg
                  disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
            {uploadedFile && !isRecording && (
              <div className="mt-4 p-3 bg-blue-50 rounded-xl border border-blue-100">
                <p className="text-sm text-gray-700">
                  <span className="font-semibold text-blue-700">Selected:</span> {uploadedFile.name}
                </p>
              </div>
            )}
          </div>

          {/* Record Audio */}
          <div className="group bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
            <div className="flex items-center gap-3 mb-6">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center shadow-lg transition-all ${
                isRecording 
                  ? 'bg-gradient-to-br from-red-500 to-pink-600 animate-pulse' 
                  : 'bg-gradient-to-br from-green-500 to-emerald-600'
              }`}>
                <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                </svg>
              </div>
              <div>
                <label className="font-bold text-gray-800 text-lg block">
                  Record Audio
                </label>
                {isRecording && (
                  <span className="text-xs text-red-600 font-medium animate-pulse">
                    Recording... {formatTime(recordingTime)}
                  </span>
                )}
              </div>
            </div>
            
            {/* Audio Source Selection */}
            {!isRecording && (
              <div className="mb-5 flex gap-3">
                <button
                  onClick={() => setRecordingSource('microphone')}
                  disabled={isProcessing}
                  className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all duration-300 ${
                    recordingSource === 'microphone'
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:scale-105'
                  } disabled:opacity-50`}
                >
                  <span className="text-lg mr-2">🎤</span>
                  Microphone
                </button>
                <button
                  onClick={() => setRecordingSource('system')}
                  disabled={isProcessing}
                  className={`flex-1 py-3 px-4 rounded-xl font-semibold transition-all duration-300 ${
                    recordingSource === 'system'
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:scale-105'
                  } disabled:opacity-50`}
                >
                  <span className="text-lg mr-2">🔊</span>
                  System Audio
                </button>
              </div>
            )}
            
            <div className="flex flex-col gap-4">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`py-4 px-6 rounded-xl font-bold text-white transition-all duration-300 shadow-lg
                  ${isRecording 
                    ? 'bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700 hover:shadow-xl hover:scale-105' 
                    : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 hover:shadow-xl hover:scale-105'
                  } disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
                  flex items-center justify-center gap-3`}
              >
                {isRecording ? (
                  <>
                    <span className="w-4 h-4 bg-white rounded-sm"></span>
                    Stop Recording
                  </>
                ) : (
                  <>
                    <span className="w-4 h-4 bg-white rounded-full"></span>
                    Start Recording
                  </>
                )}
              </button>
              <p className="text-xs text-gray-500 text-center leading-relaxed">
                {recordingSource === 'microphone' 
                  ? '🎙️ Record audio from your microphone'
                  : '💻 Capture audio playing on your device'}
              </p>
            </div>
          </div>
        </div>

        {/* Processing Mode Selector */}
        {uploadedFilename && !isProcessing && (
          <div className="mb-6">
            <label className="block text-center font-bold text-gray-800 mb-4 text-lg">
              🎯 Select Processing Mode
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Flash Mode */}
              <button
                onClick={() => setProcessingMode('flash')}
                className={`group relative p-6 rounded-2xl border-2 transition-all duration-300 hover:scale-105 ${
                  processingMode === 'flash'
                    ? 'border-orange-500 bg-gradient-to-br from-orange-50 to-red-50 shadow-xl'
                    : 'border-gray-200 bg-white hover:border-orange-300 hover:shadow-lg'
                }`}
              >
                <div className="text-center">
                  <div className={`w-16 h-16 rounded-xl mx-auto mb-3 flex items-center justify-center transition-all ${
                    processingMode === 'flash'
                      ? 'bg-gradient-to-br from-orange-500 to-red-500 scale-110'
                      : 'bg-gradient-to-br from-orange-400 to-red-400 group-hover:scale-110'
                  }`}>
                    <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Flash Mode</h3>
                  <p className="text-sm text-gray-600 mb-2">⚡ Fastest Processing</p>
                  <p className="text-xs text-gray-500">Quick results for short meetings</p>
                  <div className="mt-3 flex items-center justify-center gap-2 text-xs text-gray-600">
                    <span className="px-2 py-1 bg-orange-100 rounded">Base Model</span>
                  </div>
                </div>
              </button>

              {/* Flow Mode */}
              <button
                onClick={() => setProcessingMode('flow')}
                className={`group relative p-6 rounded-2xl border-2 transition-all duration-300 hover:scale-105 ${
                  processingMode === 'flow'
                    ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-indigo-50 shadow-xl'
                    : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-lg'
                }`}
              >
                <div className="text-center">
                  <div className={`w-16 h-16 rounded-xl mx-auto mb-3 flex items-center justify-center transition-all ${
                    processingMode === 'flow'
                      ? 'bg-gradient-to-br from-blue-500 to-indigo-600 scale-110'
                      : 'bg-gradient-to-br from-blue-400 to-indigo-500 group-hover:scale-110'
                  }`}>
                    <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Flow Mode</h3>
                  <p className="text-sm text-gray-600 mb-2">⚖️ Balanced Quality</p>
                  <p className="text-xs text-gray-500">Recommended for most meetings</p>
                  <div className="mt-3 flex items-center justify-center gap-2 text-xs text-gray-600">
                    <span className="px-2 py-1 bg-blue-100 rounded">Small Model</span>
                  </div>
                </div>
              </button>

              {/* Deep Mode */}
              <button
                onClick={() => setProcessingMode('deep')}
                className={`group relative p-6 rounded-2xl border-2 transition-all duration-300 hover:scale-105 ${
                  processingMode === 'deep'
                    ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-pink-50 shadow-xl'
                    : 'border-gray-200 bg-white hover:border-purple-300 hover:shadow-lg'
                }`}
              >
                <div className="text-center">
                  <div className={`w-16 h-16 rounded-xl mx-auto mb-3 flex items-center justify-center transition-all ${
                    processingMode === 'deep'
                      ? 'bg-gradient-to-br from-purple-500 to-pink-600 scale-110'
                      : 'bg-gradient-to-br from-purple-400 to-pink-500 group-hover:scale-110'
                  }`}>
                    <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                      <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm9.707 5.707a1 1 0 00-1.414-1.414L9 12.586l-1.293-1.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <h3 className="font-bold text-lg text-gray-800 mb-2">Deep Mode</h3>
                  <p className="text-sm text-gray-600 mb-2">🎯 Highest Accuracy</p>
                  <p className="text-xs text-gray-500">Detailed analysis for important meetings</p>
                  <div className="mt-3 flex items-center justify-center gap-2 text-xs text-gray-600">
                    <span className="px-2 py-1 bg-purple-100 rounded">Medium Model</span>
                  </div>
                </div>
              </button>
            </div>
          </div>
        )}

        {/* Run Processing Button */}
        {uploadedFilename && !isProcessing && (
          <div className="mb-8">
            <button
              onClick={handleRunProcessing}
              className="group relative w-full py-7 bg-gradient-to-r from-orange-500 via-red-500 to-pink-500 
                       hover:from-orange-600 hover:via-red-600 hover:to-pink-600
                       text-white font-bold text-xl rounded-2xl shadow-2xl 
                       transition-all duration-300 transform hover:scale-[1.02]
                       flex items-center justify-center gap-4 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 
                            translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
              <svg className="w-9 h-9 relative z-10" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
              </svg>
              <span className="relative z-10">
                RUN - Start AI Processing 
                {processingMode === 'flash' && ' ⚡'}
                {processingMode === 'flow' && ' ⚖️'}
                {processingMode === 'deep' && ' 🎯'}
              </span>
            </button>
            <p className="text-center text-sm text-gray-600 mt-3 font-medium">
              🚀 Processing with {processingMode === 'flash' ? 'Flash' : processingMode === 'flow' ? 'Flow' : 'Deep'} mode - 
              {processingMode === 'flash' && ' Fastest speed, good quality'}
              {processingMode === 'flow' && ' Balanced speed and quality'}
              {processingMode === 'deep' && ' Best quality, detailed analysis'}
            </p>
          </div>
        )}

        {isProcessing && (
          <div className="mb-8">
            <div className="max-w-3xl mx-auto bg-white/80 backdrop-blur-sm border-2 border-blue-200 rounded-2xl p-8 shadow-2xl">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-200"></div>
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent absolute top-0"></div>
                  </div>
                  <div>
                    <h3 className="text-blue-700 font-bold text-xl">Processing Audio</h3>
                    <p className="text-blue-600 text-sm">{processingStage || 'Initializing...'}</p>
                  </div>
                </div>
                
                {/* Cancel Button */}
                <button
                  onClick={handleCancelProcessing}
                  className="px-6 py-3 bg-red-500 hover:bg-red-600 text-white rounded-xl font-semibold 
                           transition-all duration-300 hover:scale-105 shadow-lg flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Cancel
                </button>
              </div>

              {/* Progress Bar */}
              <div className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-semibold text-gray-700">Progress</span>
                  <span className="text-sm font-bold text-blue-600">{processingProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden shadow-inner">
                  <div 
                    className="bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 h-4 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
                    style={{ width: `${processingProgress}%` }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-shimmer"></div>
                  </div>
                </div>
              </div>

              {/* Time Remaining */}
              {estimatedTimeRemaining !== null && estimatedTimeRemaining > 0 && (
                <div className="flex items-center justify-center gap-2 text-gray-600 text-sm">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Estimated time remaining: <strong>{estimatedTimeRemaining}s</strong></span>
                </div>
              )}

              {/* Processing Mode Info */}
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex items-center justify-center gap-3 text-xs text-gray-500">
                  <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full font-semibold">
                    {processingMode === 'flash' ? '⚡ Flash Mode' : processingMode === 'flow' ? '⚖️ Flow Mode' : '🎯 Deep Mode'}
                  </span>
                  <span>•</span>
                  <span>{uploadedFilename}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Audio Player Section */}
        {audioFile && (
          <div className="relative bg-gradient-to-br from-indigo-600 via-blue-600 to-cyan-500 text-white rounded-2xl p-10 mb-8 shadow-2xl overflow-hidden">
            {/* Decorative background */}
            <div className="absolute inset-0 opacity-10">
              <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full blur-3xl"></div>
              <div className="absolute bottom-0 left-0 w-64 h-64 bg-white rounded-full blur-3xl"></div>
            </div>
            
            <div className="relative z-10">
              <div className="text-center mb-6">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full mb-3">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <p className="text-white/90 text-sm font-medium">
                    Audio Player Active
                  </p>
                </div>
                <div className="text-white font-mono text-2xl font-bold">
                  {formatTime(currentTime)} <span className="text-white/60">/</span> {formatTime(duration)}
                </div>
              </div>
              
              <div ref={waveformRef} className="w-full mb-8 rounded-xl bg-white/10 backdrop-blur-sm p-5 shadow-inner" />
              
              <div className="flex justify-center items-center gap-6">
                {/* Skip Backward 5s */}
                <button
                  onClick={skipBackward}
                  className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-sm hover:bg-white/30 
                           flex items-center justify-center shadow-xl
                           transition-all duration-300 transform hover:scale-110
                           border border-white/30"
                  title="Lùi 5 giây"
                >
                  <svg className="w-7 h-7 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M11.99 5V1l-5 5 5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6h-2c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8zm-1.1 11h-.85v-3.26l-1.01.31v-.69l1.77-.63h.09V16zm4.28-1.76c0 .32-.03.6-.1.82s-.17.42-.29.57-.28.26-.45.33-.37.1-.59.1-.41-.03-.59-.1-.33-.18-.46-.33-.23-.34-.3-.57-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82s.17-.42.29-.57.28-.26.45-.33.37-.1.59-.1.41.03.59.1.33.18.46.33.23.34.3.57.11.5.11.82v.74zm-.85-.86c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31-.11-.14-.19-.17-.16-.05-.25-.05-.18.02-.25.05-.14.09-.19.17-.09.18-.12.31-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32.11.14.19.17.16.05.25.05.18-.02.25-.05.14-.09.19-.17.09-.19.11-.32.04-.29.04-.48v-.97z"/>
                  </svg>
                </button>

                {/* Play/Pause Button */}
                <button
                  onClick={togglePlay}
                  className="w-24 h-24 rounded-full bg-white text-indigo-600
                           flex items-center justify-center shadow-2xl
                           transition-all duration-300 transform hover:scale-110
                           hover:shadow-white/50"
                >
                  <svg 
                    className="w-10 h-10" 
                    fill="currentColor" 
                    viewBox="0 0 20 20"
                  >
                    {isPlaying ? (
                      <>
                        <rect x="5" y="3" width="3" height="14" rx="1" />
                        <rect x="12" y="3" width="3" height="14" rx="1" />
                      </>
                    ) : (
                      <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                    )}
                  </svg>
                </button>

                {/* Skip Forward 5s */}
                <button
                  onClick={skipForward}
                  className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-sm hover:bg-white/30 
                           flex items-center justify-center shadow-xl
                           transition-all duration-300 transform hover:scale-110
                           border border-white/30"
                  title="Tiến 5 giây"
                >
                  <svg className="w-7 h-7 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8zm-.36 11h-.85v-3.26l-1.01.31v-.69l1.77-.63h.09V16zm4.28-1.76c0 .32-.03.6-.1.82s-.17.42-.29.57-.28.26-.45.33-.37.1-.59.1-.41-.03-.59-.1-.33-.18-.46-.33-.23-.34-.3-.57-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82s.17-.42.29-.57.28-.26.45-.33.37-.1.59-.1.41.03.59.1.33.18.46.33.23.34.3.57.11.5.11.82v.74zm-.85-.86c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31-.11-.14-.19-.17-.16-.05-.25-.05-.18.02-.25.05-.14.09-.19.17-.09.18-.12.31-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32.11.14.19.17.16.05.25.05.18-.02.25-.05.14-.09.19-.17.09-.19.11-.32.04-.29.04-.48v-.97z"/>
                  </svg>
                </button>
              </div>

              {/* Download Audio Button */}
              <div className="mt-8 text-center">
                <button
                  onClick={downloadAudio}
                  className="px-8 py-3 bg-white/20 backdrop-blur-sm hover:bg-white/30 text-white rounded-xl 
                           transition-all duration-300 flex items-center gap-3 mx-auto
                           border border-white/30 hover:scale-105 font-semibold shadow-lg"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download Audio
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Content Sections - Vertical Layout */}
        <div className="space-y-8">
          {/* Dialog/Transcript */}
          <div className="group bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 overflow-hidden">
            <div className="bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-600 text-white px-8 py-5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <h2 className="font-bold text-xl">
                    Dialog / Transcript
                  </h2>
                  {transcript.length > 0 && (
                    <span className="text-xs text-blue-100">
                      {transcript.length} segments {audioFile && '• Click timestamp to jump ⏯️'}
                    </span>
                  )}
                </div>
              </div>
              {transcript.length > 0 && (
                <button
                  onClick={downloadDialog}
                  className="px-5 py-2.5 bg-white/20 backdrop-blur-sm hover:bg-white/30 rounded-xl transition-all duration-300 flex items-center gap-2 font-semibold hover:scale-105"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              )}
            </div>
            
            {/* Search Bar */}
            {transcript.length > 0 && (
              <div className="px-8 pt-4 pb-2 border-b border-gray-200">
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <input
                    type="text"
                    placeholder="Search transcript..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-10 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-gray-700"
                  />
                  {searchQuery && (
                    <button
                      onClick={() => setSearchQuery('')}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
                      title="Clear search"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            )}
            
            <div className="p-8 max-h-[800px] overflow-y-auto">
              {transcript.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-10 h-10 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <p className="text-gray-400 font-medium">
                    Upload and process an audio file to see the transcript
                  </p>
                </div>
              ) : (
                <div className="space-y-5">
                  {filteredTranscript.length === 0 ? (
                    <div className="text-center py-12">
                      <svg className="w-16 h-16 text-gray-300 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      <p className="text-gray-500 font-medium">No results found for "{searchQuery}"</p>
                      <button
                        onClick={() => setSearchQuery('')}
                        className="mt-3 text-sm text-blue-600 hover:text-blue-700 font-medium"
                      >
                        Clear search
                      </button>
                    </div>
                  ) : (
                    <>
                      {searchQuery && (
                        <div className="text-sm text-gray-600 mb-4 px-2">
                          Found {filteredTranscript.length} of {transcript.length} segments
                        </div>
                      )}
                      {filteredTranscript.map((line, index) => {
                        // Find original index for proper highlighting
                        const originalIndex = transcript.findIndex(t => t === line);
                        
                        return (
                          <div 
                            key={originalIndex} 
                            className={`group/item border-l-4 pl-5 py-3 rounded-r-lg transition-all duration-200 cursor-pointer ${
                              activeSegmentIndex === originalIndex
                                ? 'border-green-500 bg-gradient-to-r from-green-50 to-green-100 shadow-md scale-[1.02]'
                                : 'border-blue-400 hover:bg-gradient-to-r hover:from-blue-50 hover:to-transparent hover:border-blue-500'
                            }`}
                            onClick={() => handleTimestampClick(line.time, originalIndex)}
                          >
                            <div className="flex items-center gap-3 mb-2">
                              <button
                                className={`px-3 py-1.5 font-semibold text-xs rounded-lg transition-all hover:scale-110 ${
                                  activeSegmentIndex === originalIndex
                                    ? 'bg-green-500 text-white shadow-md'
                                    : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                }`}
                                title="Click to jump to this timestamp"
                              >
                                {audioFile && '▶ '}{line.time}
                              </button>
                              <span 
                                className="font-bold text-gray-800 text-base"
                                dangerouslySetInnerHTML={{ __html: highlightText(line.speaker, searchQuery) }}
                              />
                            </div>
                            <p 
                              className="text-gray-700 leading-relaxed"
                              dangerouslySetInnerHTML={{ __html: highlightText(line.text, searchQuery) }}
                            />
                          </div>
                        );
                      })}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Summary */}
          <div className="group bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 overflow-hidden">
            <div className="bg-gradient-to-r from-purple-500 via-purple-600 to-pink-600 text-white px-8 py-5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                    <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                  </svg>
                </div>
                <h2 className="font-bold text-xl">
                  Summary
                </h2>
              </div>
              {summary && (
                <button
                  onClick={downloadSummary}
                  className="px-5 py-2.5 bg-white/20 backdrop-blur-sm hover:bg-white/30 rounded-xl transition-all duration-300 flex items-center gap-2 font-semibold hover:scale-105"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              )}
            </div>
            <div className="p-8 max-h-96 overflow-y-auto">
              {summary ? (
                <div 
                  className="text-gray-700 leading-relaxed prose prose-sm max-w-none"
                  dangerouslySetInnerHTML={{ __html: formatSummary(summary) }}
                />
              ) : (
                <div className="text-center py-16">
                  <div className="w-20 h-20 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-10 h-10 text-purple-500" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                      <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <p className="text-gray-400 font-medium">
                    Summary will appear here after processing
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Tasks */}
          <div className="group bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl shadow-xl hover:shadow-2xl transition-all duration-300 overflow-hidden">
            <div className="bg-gradient-to-r from-green-500 via-emerald-600 to-teal-600 text-white px-8 py-5 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd" />
                  </svg>
                </div>
                <h2 className="font-bold text-xl">
                  Next Actions / Tasks
                </h2>
              </div>
              {tasks.length > 0 && (
                <button
                  onClick={downloadTasks}
                  className="px-5 py-2.5 bg-white/20 backdrop-blur-sm hover:bg-white/30 rounded-xl transition-all duration-300 flex items-center gap-2 font-semibold hover:scale-105"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              )}
            </div>
            <div className="p-8 max-h-96 overflow-y-auto">
              {tasks.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-20 h-20 bg-gradient-to-br from-green-100 to-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-10 h-10 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <p className="text-gray-400 font-medium">
                    Tasks will be extracted after processing
                  </p>
                </div>
              ) : (
                <ul className="space-y-4">
                  {tasks.map((task, index) => {
                    const isObject = typeof task === 'object' && task !== null;
                    
                    // Handle task text
                    let taskText = '';
                    let howToText = '';
                    
                    if (typeof task === 'string') {
                      taskText = task;
                    } else if (isObject) {
                      taskText = task.task || '';
                      howToText = task.how_to || '';
                      // Skip empty tasks
                      if (!taskText.trim() && !howToText.trim()) return null;
                    }
                    
                    const assignedTo = isObject ? task.assigned_to : null;
                    const deadline = isObject ? task.deadline : null;
                    const priority = isObject ? task.priority : null;
                    
                    return (
                      <li key={index} className="group/item flex items-start gap-4 p-5 hover:bg-gradient-to-r hover:from-green-50 hover:to-transparent transition-all duration-200 rounded-xl border border-gray-200 hover:border-green-300 hover:shadow-md">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center flex-shrink-0 mt-1 shadow-md">
                          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <p className="text-gray-800 font-semibold mb-2 text-base">{taskText}</p>
                          {howToText && (
                            <p className="text-gray-600 text-sm mb-3 pl-4 border-l-2 border-blue-300 italic bg-blue-50/50 py-2 rounded-r">
                              💡 {howToText}
                            </p>
                          )}
                          <div className="flex flex-wrap gap-2">
                            {assignedTo && (
                              <span className="flex items-center gap-1.5 px-3 py-1 bg-blue-100 text-blue-700 rounded-lg text-sm font-medium">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                </svg>
                                {assignedTo}
                              </span>
                            )}
                            {deadline && (
                              <span className="flex items-center gap-1.5 px-3 py-1 bg-orange-100 text-orange-700 rounded-lg text-sm font-medium">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                {deadline}
                              </span>
                            )}
                            {priority && (
                              <span className={`px-3 py-1 rounded-lg text-sm font-bold uppercase ${
                                priority === 'high' ? 'bg-red-100 text-red-700' :
                                priority === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                'bg-green-100 text-green-700'
                              }`}>
                                {priority}
                              </span>
                            )}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>
        </div>

        {/* Download All Button */}
        {(transcript.length > 0 || summary || tasks.length > 0) && (
          <div className="mt-10 text-center">
            <button
              onClick={downloadAll}
              className="group relative px-10 py-5 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 
                       hover:from-blue-700 hover:via-indigo-700 hover:to-purple-700
                       text-white font-bold rounded-2xl shadow-2xl 
                       transition-all duration-300 transform hover:scale-105
                       flex items-center gap-4 mx-auto overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 
                            translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
              <svg className="w-7 h-7 relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span className="relative z-10 text-lg">Download Complete Report</span>
            </button>
            <p className="text-gray-600 text-sm mt-3 font-medium">
              Get all transcript, summary and tasks in one file
            </p>
          </div>
        )}

        {/* Bottom Info Section */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="group bg-white/80 backdrop-blur-sm rounded-2xl p-8 border border-gray-200/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <svg className="w-7 h-7 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
              </svg>
            </div>
            <h3 className="font-bold text-gray-800 mb-3 text-lg">
              Online Voice Recorder
            </h3>
            <p className="text-gray-600 leading-relaxed">
              Our Voice Recorder is a convenient and simple online tool that can be used right in your browser.
            </p>
          </div>
          <div className="group bg-white/80 backdrop-blur-sm rounded-2xl p-8 border border-gray-200/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <svg className="w-7 h-7 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" clipRule="evenodd" />
              </svg>
            </div>
            <h3 className="font-bold text-gray-800 mb-3 text-lg">Free to use</h3>
            <p className="text-gray-600 leading-relaxed">
              Voice Recorder is completely free. No hidden payments, activation fees, or charges for extra features.
            </p>
          </div>
          <div className="group bg-white/80 backdrop-blur-sm rounded-2xl p-8 border border-gray-200/50 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
            <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
              <svg className="w-7 h-7 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
              </svg>
            </div>
            <h3 className="font-bold text-gray-800 mb-3 text-lg">
              AI-Powered Analysis
            </h3>
            <p className="text-gray-600 leading-relaxed">
              Advanced AI technology for accurate transcription, intelligent summarization and automatic task extraction.
            </p>
          </div>
        </div>

        {/* Processing History */}
        {processingHistory.length > 0 && (
          <div className="mt-12">
            <div className="bg-white/80 backdrop-blur-sm border border-gray-200/50 rounded-2xl shadow-xl overflow-hidden">
              <div className="bg-gradient-to-r from-gray-700 via-gray-800 to-gray-900 text-white px-8 py-5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur-sm flex items-center justify-center">
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div>
                    <h2 className="font-bold text-xl">Processing History</h2>
                    <span className="text-xs text-gray-300">Recent processing sessions</span>
                  </div>
                </div>
              </div>
              <div className="p-6 max-h-96 overflow-y-auto">
                <div className="space-y-3">
                  {processingHistory.map((item, index) => (
                    <div 
                      key={item.id}
                      className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl hover:from-blue-50 hover:to-indigo-50 transition-all duration-200 border border-gray-200"
                    >
                      <div className="flex items-center gap-4 flex-1">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold">
                          {index + 1}
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-800 text-sm truncate">{item.filename}</h4>
                          <div className="flex items-center gap-3 mt-1">
                            <span className="text-xs text-gray-600">
                              {new Date(item.timestamp).toLocaleString('vi-VN', {
                                year: 'numeric',
                                month: '2-digit',
                                day: '2-digit',
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </span>
                            <span className="text-xs text-gray-400">•</span>
                            <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full font-medium">
                              {item.mode === 'flash' ? '⚡ Flash' : item.mode === 'flow' ? '⚖️ Flow' : '🎯 Deep'}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <div className="text-sm font-bold text-green-600">{item.processingTime}s</div>
                          <div className="text-xs text-gray-500">{item.segmentCount} segments</div>
                        </div>
                        <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                          <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
            padding: '16px',
            borderRadius: '10px',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10B981',
              secondary: '#fff',
            },
          },
          error: {
            duration: 4000,
            iconTheme: {
              primary: '#EF4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </div>
  );
}
