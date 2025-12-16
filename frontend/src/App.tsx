import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Upload,
  Send,
  File,
  Trash2,
  Download,
  ChevronRight,
  Loader,
  AlertCircle,
} from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'node_info';
  content: string;
  node?: string;
  timestamp: Date;
}

interface Document {
  filename: string;
  size: number;
  upload_time: string;
}

interface NodeInfo {
  type: 'enter' | 'exit';
  node: string;
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/chat';

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [showDocuments, setShowDocuments] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [nodeStack, setNodeStack] = useState<string[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/documents`);
      if (response.ok) {
        const docs = await response.json();
        setDocuments(docs);
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file.name.endsWith('.pdf')) {
      setError('Only PDF files are allowed');
      return;
    }

    setUploadingFile(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        await loadDocuments();
        setMessages((prev) => [
          ...prev,
          {
            id: Math.random().toString(),
            type: 'node_info',
            content: `📄 Document "${file.name}" uploaded successfully`,
            timestamp: new Date(),
          },
        ]);
      } else {
        setError('Failed to upload document');
      }
    } catch (err) {
      setError('Upload failed: ' + (err as Error).message);
    } finally {
      setUploadingFile(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDownloadDocument = async (filename: string) => {
    try {
      const response = await fetch(`${API_URL}/documents/${filename}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      setError('Download failed: ' + (err as Error).message);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    if (!window.confirm(`Delete "${filename}"?`)) return;

    try {
      const response = await fetch(`${API_URL}/documents/${filename}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        await loadDocuments();
        setMessages((prev) => [
          ...prev,
          {
            id: Math.random().toString(),
            type: 'node_info',
            content: `🗑️ Document "${filename}" deleted`,
            timestamp: new Date(),
          },
        ]);
      }
    } catch (err) {
      setError('Delete failed: ' + (err as Error).message);
    }
  };

  const connectWebSocket = () => {
    return new Promise<WebSocket>((resolve, reject) => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('WebSocket connected');
        resolve(ws);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
    });
  };

  const handleSendMessage = useCallback(async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput) return;

    setInput('');
    setLoading(true);
    setError(null);
    setNodeStack([]);

    // Add user message
    const userMessageId = Math.random().toString();
    setMessages((prev) => [
      ...prev,
      {
        id: userMessageId,
        type: 'user',
        content: trimmedInput,
        timestamp: new Date(),
      },
    ]);

    try {
      // Connect WebSocket if not already connected
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        wsRef.current = await connectWebSocket();
      }

      const ws = wsRef.current;
      let assistantContent = '';
      let assistantMessageId = Math.random().toString();

      // Send query
      ws.send(JSON.stringify({ query: trimmedInput }));

      // Handle WebSocket messages
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'node_enter') {
          setNodeStack((prev) => [...prev, data.node]);
          setMessages((prev) => [
            ...prev,
            {
              id: Math.random().toString(),
              type: 'node_info',
              content: `▶️ Entering: ${data.node}`,
              timestamp: new Date(),
            },
          ]);
        } else if (data.type === 'node_exit') {
          setNodeStack((prev) => prev.filter((n) => n !== data.node));
          setMessages((prev) => [
            ...prev,
            {
              id: Math.random().toString(),
              type: 'node_info',
              content: `◀️ Exiting: ${data.node}`,
              timestamp: new Date(),
            },
          ]);
        } else if (data.type === 'content') {
          assistantContent += data.data;

          // Update or create assistant message
          setMessages((prev) => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage?.id === assistantMessageId) {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  content: assistantContent,
                },
              ];
            } else {
              return [
                ...prev,
                {
                  id: assistantMessageId,
                  type: 'assistant',
                  content: assistantContent,
                  timestamp: new Date(),
                },
              ];
            }
          });
        } else if (data.type === 'done') {
          console.log('Query completed');
          setLoading(false);
        } else if (data.type === 'error') {
          setError(data.message);
          setLoading(false);
        }
      };

      ws.onerror = (err) => {
        setError('WebSocket error occurred');
        setLoading(false);
      };
    } catch (err) {
      setError('Failed to send message: ' + (err as Error).message);
      setLoading(false);
    }
  }, [input]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div
        className={`${
          sidebarOpen ? 'w-64' : 'w-20'
        } bg-white border-r border-gray-200 transition-all duration-300 flex flex-col`}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            {sidebarOpen && (
              <h1 className="text-lg font-bold text-gray-800">RAG Chat</h1>
            )}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg transition"
            >
              <ChevronRight
                size={20}
                className={`transform transition ${sidebarOpen ? 'rotate-180' : ''}`}
              />
            </button>
          </div>
        </div>

        {/* Upload Section */}
        {sidebarOpen && (
          <div className="p-4 border-b border-gray-200">
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              📤 Upload Document
            </label>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadingFile}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 transition flex items-center justify-center gap-2"
            >
              <Upload size={18} />
              {uploadingFile ? 'Uploading...' : 'Choose PDF'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
            />
          </div>
        )}

        {/* Documents Section */}
        {sidebarOpen && (
          <div className="flex-1 overflow-y-auto p-4">
            <button
              onClick={() => setShowDocuments(!showDocuments)}
              className="w-full text-left font-semibold text-gray-700 mb-3 flex items-center gap-2 hover:text-blue-600 transition"
            >
              <File size={18} />
              📚 Documents ({documents.length})
            </button>

            {showDocuments && (
              <div className="space-y-2">
                {documents.length === 0 ? (
                  <p className="text-sm text-gray-500 italic">No documents</p>
                ) : (
                  documents.map((doc) => (
                    <div
                      key={doc.filename}
                      className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition"
                    >
                      <p className="text-sm font-medium text-gray-800 truncate">
                        {doc.filename}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {(doc.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                      <div className="flex gap-2 mt-2">
                        <button
                          onClick={() => handleDownloadDocument(doc.filename)}
                          className="flex-1 text-xs bg-green-50 text-green-700 px-2 py-1 rounded hover:bg-green-100 transition flex items-center justify-center gap-1"
                        >
                          <Download size={14} />
                          Download
                        </button>
                        <button
                          onClick={() => handleDeleteDocument(doc.filename)}
                          className="flex-1 text-xs bg-red-50 text-red-700 px-2 py-1 rounded hover:bg-red-100 transition flex items-center justify-center gap-1"
                        >
                          <Trash2 size={14} />
                          Delete
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <h2 className="text-xl font-bold text-gray-800">
            Multi-Agent RAG Chat
          </h2>
          <p className="text-sm text-gray-600">
            Ask questions about your uploaded documents
          </p>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <h3 className="text-2xl font-bold text-gray-800 mb-2">
                  Welcome to RAG Chat
                </h3>
                <p className="text-gray-600">
                  Upload documents and start asking questions
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-2xl px-4 py-3 rounded-lg ${
                    msg.type === 'user'
                      ? 'bg-blue-500 text-white rounded-br-none'
                      : msg.type === 'node_info'
                        ? 'bg-gray-200 text-gray-700 rounded-bl-none text-sm italic'
                        : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'
                  }`}
                >
                  <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                  <p
                    className={`text-xs mt-2 ${
                      msg.type === 'user' ? 'text-blue-100' : 'text-gray-500'
                    }`}
                  >
                    {msg.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 rounded-lg rounded-bl-none px-4 py-3 flex items-center gap-2">
                <Loader size={18} className="animate-spin text-blue-500" />
                <span className="text-gray-600">Processing...</span>
              </div>
            </div>
          )}

          {nodeStack.length > 0 && (
            <div className="flex justify-start">
              <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2">
                <p className="text-xs text-amber-800 font-semibold mb-1">
                  Active Nodes:
                </p>
                <div className="space-y-1">
                  {nodeStack.map((node) => (
                    <p key={node} className="text-xs text-amber-700">
                      ⚙️ {node}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-t border-red-200 px-6 py-3 flex items-center gap-2">
            <AlertCircle size={18} className="text-red-600" />
            <p className="text-red-700">{error}</p>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-600 hover:text-red-800 font-semibold"
            >
              ✕
            </button>
          </div>
        )}

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex gap-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
              placeholder="Type your question here... (Shift+Enter for new line)"
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none disabled:bg-gray-100"
              rows={3}
            />
            <button
              onClick={handleSendMessage}
              disabled={loading || !input.trim()}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 transition flex items-center justify-center"
            >
              {loading ? (
                <Loader size={20} className="animate-spin" />
              ) : (
                <Send size={20} />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
