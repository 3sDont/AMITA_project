import { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";

export default function App() {
  const [audioFile, setAudioFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [uploadedFilename, setUploadedFilename] = useState(null); // Store backend filename
  const waveformRef = useRef(null);
  const wavesurfer = useRef(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const recordingInterval = useRef(null);

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
      alert("Upload failed. Make sure backend server is running!");
    }
  };

  const handleRunProcessing = async () => {
    if (!uploadedFilename) {
      alert("Please upload or record an audio file first!");
      return;
    }
    console.log("Processing file:", uploadedFilename);
    await processAudio(uploadedFilename);
  };

  const processAudio = async (filename) => {
    setIsProcessing(true);
    setTranscript([{ time: "00:00", speaker: "System", text: "Processing audio... Please wait..." }]);
    setSummary("Processing...");
    setTasks(["Audio processing in progress..."]);

    try {
      const response = await fetch(`${API_URL}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("✅ Processing response:", data);
        console.log("📝 Transcript:", data.transcript?.length, "items");
        console.log("📄 Summary:", data.summary?.substring(0, 50));
        console.log("✓ Tasks:", data.tasks?.length, "items");
        setTranscript(data.transcript);
        setSummary(data.summary);
        setTasks(data.tasks);
      } else {
        throw new Error("Processing failed");
      }
    } catch (error) {
      console.error("Processing failed:", error);
      setTranscript([{ time: "00:00", speaker: "Error", text: "Processing failed. Check backend server." }]);
      setSummary("Error occurred during processing.");
      setTasks(["Please try again"]);
    } finally {
      setIsProcessing(false);
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
      });

      wavesurfer.current.on('finish', () => {
        setIsPlaying(false);
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

  // Recording functions
  const startRecording = async () => {
    try {
      // Clear previous data
      setUploadedFilename(null);
      setTranscript([]);
      setSummary("");
      setTasks([]);
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
      alert('Could not access microphone. Please grant permission.');
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
      alert("Upload failed. Make sure backend server is running!");
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
    <div className="min-h-screen bg-white">
      {/* Header Section with Navy Blue Background */}
      <div className="bg-gradient-to-r from-[#1e3a8a] to-[#1e40af] text-white py-16 px-4">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-5xl font-bold mb-3">AMITA</h1>
          <p className="text-xl text-blue-100">
            Meeting Insight & Task Assistant
          </p>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Upload and Record Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Upload Audio */}
          <div className="bg-white border-2 border-gray-200 rounded-xl p-8 shadow-sm">
            <label className="flex items-center gap-2 font-semibold text-gray-800 mb-4 text-lg">
              <span className="text-2xl">📁</span>
              Upload Audio File
              {isProcessing && (
                <span className="text-sm text-blue-600 font-normal ml-2 animate-pulse">
                  Processing...
                </span>
              )}
            </label>
            <div className="relative">
              <input
                type="file"
                accept="audio/*"
                onChange={handleUpload}
                disabled={isProcessing || isRecording}
                className="block w-full text-sm text-gray-600
                  file:mr-4 file:py-3 file:px-6
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  file:cursor-pointer cursor-pointer
                  file:transition-colors
                  disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>
            {uploadedFile && !isRecording && (
              <p className="mt-3 text-sm text-gray-600">
                Selected: <span className="font-medium">{uploadedFile.name}</span>
              </p>
            )}
          </div>

          {/* Record Audio */}
          <div className="bg-white border-2 border-gray-200 rounded-xl p-8 shadow-sm">
            <label className="flex items-center gap-2 font-semibold text-gray-800 mb-4 text-lg">
              <span className="text-2xl">🎙️</span>
              Record Audio
              {isRecording && (
                <span className="text-sm text-red-600 font-normal ml-2 animate-pulse">
                  Recording... {formatTime(recordingTime)}
                </span>
              )}
            </label>
            <div className="flex flex-col gap-4">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`py-4 px-6 rounded-lg font-semibold text-white transition-all
                  ${isRecording 
                    ? 'bg-red-600 hover:bg-red-700 animate-pulse' 
                    : 'bg-green-600 hover:bg-green-700'
                  } disabled:opacity-50 disabled:cursor-not-allowed
                  flex items-center justify-center gap-2`}
              >
                {isRecording ? (
                  <>
                    <span className="w-4 h-4 bg-white rounded-full animate-pulse"></span>
                    Stop Recording
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                    </svg>
                    Start Recording
                  </>
                )}
              </button>
              <p className="text-xs text-gray-500 text-center">
                Click to record audio directly from your microphone
              </p>
            </div>
          </div>
        </div>

        {/* Run Processing Button */}
        {uploadedFilename && !isProcessing && (
          <div className="mb-6">
            <button
              onClick={handleRunProcessing}
              className="w-full py-6 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600
                       text-white font-bold text-xl rounded-xl shadow-xl transition-all transform hover:scale-[1.02]
                       flex items-center justify-center gap-3"
            >
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
              </svg>
              RUN - Start Processing Audio
            </button>
            <p className="text-center text-sm text-gray-600 mt-2">
              Click to analyze your audio and extract transcript, summary, and tasks
            </p>
          </div>
        )}

        {isProcessing && (
          <div className="mb-6 text-center">
            <div className="inline-flex items-center gap-3 px-6 py-4 bg-blue-50 border-2 border-blue-200 rounded-xl">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span className="text-blue-600 font-semibold">Processing your audio... Please wait</span>
            </div>
          </div>
        )}

        {/* Audio Player Section */}
        {audioFile && (
          <div className="bg-gradient-to-br from-[#1e3a8a] to-[#2563eb] text-white rounded-xl p-10 mb-6 shadow-lg">
            <div className="text-center mb-4">
              <p className="text-blue-200 text-sm mb-2">
                Audio Player
              </p>
              <div className="text-white font-mono text-lg">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>
            
            <div ref={waveformRef} className="w-full mb-6 rounded-lg bg-white/10 p-4" />
            
            <div className="flex justify-center items-center gap-4">
              {/* Skip Backward 5s */}
              <button
                onClick={skipBackward}
                className="w-14 h-14 rounded-full bg-blue-600 hover:bg-blue-700 
                         flex items-center justify-center shadow-lg
                         transition-all duration-200 transform hover:scale-105"
                title="Lùi 5 giây"
              >
                <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M11.99 5V1l-5 5 5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6h-2c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8zm-1.1 11h-.85v-3.26l-1.01.31v-.69l1.77-.63h.09V16zm4.28-1.76c0 .32-.03.6-.1.82s-.17.42-.29.57-.28.26-.45.33-.37.1-.59.1-.41-.03-.59-.1-.33-.18-.46-.33-.23-.34-.3-.57-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82s.17-.42.29-.57.28-.26.45-.33.37-.1.59-.1.41.03.59.1.33.18.46.33.23.34.3.57.11.5.11.82v.74zm-.85-.86c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31-.11-.14-.19-.17-.16-.05-.25-.05-.18.02-.25.05-.14.09-.19.17-.09.18-.12.31-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32.11.14.19.17.16.05.25.05.18-.02.25-.05.14-.09.19-.17.09-.19.11-.32.04-.29.04-.48v-.97z"/>
                </svg>
              </button>

              {/* Play/Pause Button */}
              <button
                onClick={togglePlay}
                className="w-20 h-20 rounded-full bg-red-500 hover:bg-red-600 
                         flex items-center justify-center shadow-xl
                         transition-all duration-200 transform hover:scale-105"
              >
                <svg 
                  className="w-8 h-8 text-white" 
                  fill="currentColor" 
                  viewBox="0 0 20 20"
                >
                  {isPlaying ? (
                    <rect x="6" y="4" width="3" height="12" />
                  ) : (
                    <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                  )}
                  {isPlaying && <rect x="11" y="4" width="3" height="12" />}
                </svg>
              </button>

              {/* Skip Forward 5s */}
              <button
                onClick={skipForward}
                className="w-14 h-14 rounded-full bg-blue-600 hover:bg-blue-700 
                         flex items-center justify-center shadow-lg
                         transition-all duration-200 transform hover:scale-105"
                title="Tiến 5 giây"
              >
                <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 5V1l5 5-5 5V7c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8zm-.36 11h-.85v-3.26l-1.01.31v-.69l1.77-.63h.09V16zm4.28-1.76c0 .32-.03.6-.1.82s-.17.42-.29.57-.28.26-.45.33-.37.1-.59.1-.41-.03-.59-.1-.33-.18-.46-.33-.23-.34-.3-.57-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82s.17-.42.29-.57.28-.26.45-.33.37-.1.59-.1.41.03.59.1.33.18.46.33.23.34.3.57.11.5.11.82v.74zm-.85-.86c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31-.11-.14-.19-.17-.16-.05-.25-.05-.18.02-.25.05-.14.09-.19.17-.09.18-.12.31-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32.11.14.19.17.16.05.25.05.18-.02.25-.05.14-.09.19-.17.09-.19.11-.32.04-.29.04-.48v-.97z"/>
                </svg>
              </button>
            </div>

            {/* Download Audio Button */}
            <div className="mt-4 text-center">
              <button
                onClick={downloadAudio}
                className="px-6 py-2 bg-white/20 hover:bg-white/30 text-white rounded-lg 
                         transition-all flex items-center gap-2 mx-auto"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Audio
              </button>
            </div>
          </div>
        )}

        {/* Content Sections - Vertical Layout */}
        <div className="space-y-6">
          {/* Dialog/Transcript */}
          <div className="bg-white border-2 border-gray-200 rounded-xl shadow-sm">
            <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-6 py-4 rounded-t-xl flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-semibold text-lg">
                <span className="text-2xl">💬</span>
                Dialog / Transcript
              </h2>
              {transcript.length > 0 && (
                <button
                  onClick={downloadDialog}
                  className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-all flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              )}
            </div>
            <div className="p-6 max-h-96 overflow-y-auto">
              {transcript.length === 0 ? (
                <p className="text-gray-400 text-sm text-center py-12">
                  Upload an audio file to see the transcript
                </p>
              ) : (
                <div className="space-y-4">
                  {transcript.map((line, index) => (
                    <div key={index} className="border-l-4 border-blue-400 pl-4 py-2 hover:bg-blue-50 transition-colors">
                      <div className="flex items-center gap-3 mb-1">
                        <span className="text-blue-600 font-medium text-xs">{line.time}</span>
                        <span className="font-semibold text-gray-800">{line.speaker}</span>
                      </div>
                      <p className="text-gray-700 text-sm leading-relaxed">{line.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Summary */}
          <div className="bg-white border-2 border-gray-200 rounded-xl shadow-sm">
            <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white px-6 py-4 rounded-t-xl flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-semibold text-lg">
                <span className="text-2xl">📊</span>
                Summary
              </h2>
              {summary && (
                <button
                  onClick={downloadSummary}
                  className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-all flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              )}
            </div>
            <div className="p-6 max-h-64 overflow-y-auto">
              {summary ? (
                <div className="text-gray-700 leading-relaxed whitespace-pre-wrap">
                  {summary}
                </div>
              ) : (
                <p className="text-gray-400 text-sm text-center py-12">
                  Summary will appear here after processing
                </p>
              )}
            </div>
          </div>

          {/* Tasks */}
          <div className="bg-white border-2 border-gray-200 rounded-xl shadow-sm">
            <div className="bg-gradient-to-r from-green-500 to-green-600 text-white px-6 py-4 rounded-t-xl flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-semibold text-lg">
                <span className="text-2xl">✅</span>
                Next Actions / Tasks
              </h2>
              {tasks.length > 0 && (
                <button
                  onClick={downloadTasks}
                  className="px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-all flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </button>
              )}
            </div>
            <div className="p-6 max-h-64 overflow-y-auto">
              {tasks.length === 0 ? (
                <p className="text-gray-400 text-sm text-center py-12">
                  Tasks will be extracted after processing
                </p>
              ) : (
                <ul className="space-y-3">
                  {tasks.map((task, index) => {
                    const isObject = typeof task === 'object' && task !== null;
                    
                    // Handle task text
                    let taskText = '';
                    if (typeof task === 'string') {
                      taskText = task;
                    } else if (isObject) {
                      taskText = task.task || task.how_to || '';
                      // Skip empty tasks
                      if (!taskText.trim()) return null;
                    }
                    
                    const assignedTo = isObject ? task.assigned_to : null;
                    const deadline = isObject ? task.deadline : null;
                    const priority = isObject ? task.priority : null;
                    
                    return (
                      <li key={index} className="flex items-start gap-3 p-4 hover:bg-green-50 rounded-lg transition-colors border border-gray-100">
                        <span className="text-green-600 text-xl mt-0.5">✓</span>
                        <div className="flex-1">
                          <p className="text-gray-700 font-medium mb-2">{taskText}</p>
                          <div className="flex flex-wrap gap-3 text-sm">
                            {assignedTo && (
                              <span className="flex items-center gap-1.5 text-blue-600">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                </svg>
                                {assignedTo}
                              </span>
                            )}
                            {deadline && (
                              <span className="flex items-center gap-1.5 text-orange-600">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                {deadline}
                              </span>
                            )}
                            {priority && (
                              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
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
          <div className="mt-8 text-center">
            <button
              onClick={downloadAll}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 
                       text-white font-semibold rounded-xl shadow-lg transition-all transform hover:scale-105
                       flex items-center gap-3 mx-auto"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download Complete Report
            </button>
          </div>
        )}

        {/* Bottom Info Section */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
          <div className="bg-gray-50 rounded-xl p-6">
            <h3 className="font-semibold text-gray-800 mb-2">
              Online Voice Recorder
            </h3>
            <p className="text-sm text-gray-600">
              Our Voice Recorder is a convenient and simple online tool that can be used right in your browser.
            </p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6">
            <h3 className="font-semibold text-gray-800 mb-2">Free to use</h3>
            <p className="text-sm text-gray-600">
              Voice Recorder is completely free. No hidden payments, activation fees, or charges for extra features.
            </p>
          </div>
          <div className="bg-gray-50 rounded-xl p-6">
            <h3 className="font-semibold text-gray-800 mb-2">
              Microphone settings
            </h3>
            <p className="text-sm text-gray-600">
              You can adjust your microphone settings using standard Adobe Flash Player tools.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
