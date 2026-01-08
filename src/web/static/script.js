class SGASWebClient {
    constructor() {
        this.apiUrl = 'http://localhost:8080';
        this.sessionId = null;
        this.messageCount = 0;
        this.chatHistory = [];
        this.uploadedFilesSession = [];
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    async initialize() {
        console.log('🚀 Initializing S-GAS Web Client...');
        this.initializeElements();
        this.bindEvents();
        await this.initializeSession();
        this.checkServerStatus();
        await this.loadUploadedFiles();
        console.log('✅ S-GAS Web Client initialized successfully');
    }

    async initializeSession() {
        try {
            console.log('📍 Creating session...');
            
            const response = await fetch(`${this.apiUrl}/api/session/new`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: 'web-user' })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: Failed to create session`);
            }
            
            const data = await response.json();
            this.sessionId = data.session_id;
            
            localStorage.setItem('sessionId', this.sessionId);
            
            console.log('✅ Session initialized:', this.sessionId);
            this.addMessage('system', `✅ Session created: ${this.sessionId}`, true);
            
        } catch (error) {
            console.error('❌ Session initialization failed:', error);
            this.addMessage('system', `❌ Failed to initialize session: ${error.message}`, true);
        }
    }

    initializeElements() {
        // Message Elements
        this.messageForm = document.getElementById('messageForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        
        // File upload elements
        this.fileInput = document.getElementById('fileInput');
        this.uploadButton = document.getElementById('uploadButton');
        this.fileList = document.getElementById('fileList');
        this.uploadedFiles = document.getElementById('uploadedFiles');
        this.refreshFilesButton = document.getElementById('refreshFiles');

        // Status elements
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');

        // Statistics elements
        this.messageCountElement = document.getElementById('messageCount');
        this.modelNameElement = document.getElementById('modelName');
        this.embeddingDimElement = document.getElementById('embeddingDim');
        this.requestInfoElement = document.getElementById('requestInfo');

        // Action buttons
        this.clearChatButton = document.getElementById('clearChat');
        this.exportChatButton = document.getElementById('exportChat');

        // Spinner
        this.spinner = document.getElementById('spinner');

        console.log('✅ Elements initialized');
    }

    bindEvents() {
        if (this.messageForm) {
            this.messageForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendMessage();
            });
        }

        if (this.messageInput) {
            this.messageInput.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            this.messageInput.addEventListener('input', () => {
                this.autoResizeTextarea();
            });
        }

        if (this.uploadButton && this.fileInput) {
            this.uploadButton.addEventListener('click', () => {
                this.fileInput.click();
            });

            this.fileInput.addEventListener('change', (e) => {
                this.handleFileUpload(e.target.files);
            });
        }

        if (this.messageForm) {
            this.messageForm.addEventListener('dragover', (e) => {
                e.preventDefault();
                this.messageForm.classList.add('drag-over');
            });

            this.messageForm.addEventListener('dragleave', (e) => {
                e.preventDefault();
                this.messageForm.classList.remove('drag-over');
            });

            this.messageForm.addEventListener('drop', (e) => {
                e.preventDefault();
                this.messageForm.classList.remove('drag-over');
                this.handleFileUpload(e.dataTransfer.files);
            });
        }

        if (this.refreshFilesButton) {
            this.refreshFilesButton.addEventListener('click', () => {
                this.loadUploadedFiles();
            });
        }

        if (this.clearChatButton) {
            this.clearChatButton.addEventListener('click', () => {
                this.clearChat();
            });
        }

        if (this.exportChatButton) {
            this.exportChatButton.addEventListener('click', () => {
                this.exportChat();
            });
        }

        console.log('✅ Events bound successfully');
    }

    // Separate reasoning from final answer
    separateReasoningAndAnswer(text) {
        const reasoningMarkers = [
            'Okay,', 'Alright,', 'Let me think', 'Since the context', 
            'I should', 'I need to', 'First,', 'The user'
        ];
        
        const lines = text.split('\n\n');
        let reasoningParts = [];
        let answerParts = [];
        let foundTransition = false;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            const hasReasoningMarker = reasoningMarkers.some(marker => 
                line.startsWith(marker)
            );
            
            if (!foundTransition && (hasReasoningMarker || i < lines.length - 1)) {
                reasoningParts.push(line);
            } else {
                foundTransition = true;
                answerParts.push(line);
            }
        }

        if (answerParts.length === 0) {
            return {
                reasoning: null,
                answer: text
            };
        }

        return {
            reasoning: reasoningParts.join('\n\n'),
            answer: answerParts.join('\n\n')
        };
    }

    async handleFileUpload(files) {
        if (!files || files.length === 0) return;

        if (!this.sessionId) {
            this.addMessage('system', '❌ Session not initialized', true);
            return;
        }

        const allowedTypes = ['.pdf', '.txt', '.docx', '.doc'];
        const validFiles = Array.from(files).filter(file => {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            return allowedTypes.includes(extension);
        });

        if (validFiles.length === 0) {
            this.addMessage('system', '⚠️ Supported file types: PDF, TXT, DOCX, DOC', true);
            return;
        }

        this.displaySelectedFiles(validFiles);

        for (const file of validFiles) {
            try {
                await this.uploadFile(file);
            } catch (error) {
                console.error('❌ Upload error:', error);
                this.addFileUploadMessage(file.name, 'error', error.message);
            }
        }
        // Clear the file input after upload
        this.clearFileInput();
    }

    displaySelectedFiles(files) {
        if (!this.fileList) return;
        
        this.fileList.innerHTML = '';
        this.fileList.classList.remove('hidden');
        
        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span class="file-name">${file.name}</span>
                <span class="file-size">${(file.size / 1024).toFixed(2)} KB</span>
                <span class="file-status">⏳ Uploading...</span>
            `;
            this.fileList.appendChild(fileItem);
        });
    }

    async uploadFile(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('document_type', 'general');

            this.spinner.classList.remove('hidden');

            console.log(`📤 Uploading ${file.name} to session ${this.sessionId}...`);

            const response = await fetch(
                `${this.apiUrl}/api/session/${this.sessionId}/upload-document`,
                {
                    method: 'POST',
                    body: formData
                }
            );

            this.spinner.classList.add('hidden');

            if (!response.ok) {
                throw new Error(`❌ HTTP ${response.status}: Upload failed`);
            }

            const data = await response.json();

            // Track uploaded file
            this.uploadedFilesSession.push({
                name: file.name,
                size: file.size,
                timestamp: new Date().toISOString()
            });
            
            console.log(`✅ File uploaded: ${file.name}`);
            
            this.addFileUploadMessage(
                file.name,
                'success',
                `✅ Loaded successfully! ${data.chunks_created || '?'} chunks created`
            );
        } catch (error) {
            this.spinner.classList.add('hidden');
            console.error('❌ Upload error: ', error);
            this.addFileUploadMessage(file.name, 'error', error.message);
        }

    }

    clearFileInput() {
        if (this.fileInput) {
            this.fileInput.value = '';
        }
        if (this.fileList) {
            setTimeout(() => {
                this.fileList.classList.add('hidden');
                this.fileList.innerHTML = '';
            }, 1000);
        }
    }

    async loadUploadedFiles() {
        try {
            this.spinner.classList.remove('hidden');
            
            console.log(`📁 Loading files for session ${this.sessionId}...`);
            
            const response = await fetch(
                `${this.apiUrl}/api/session/${this.sessionId}/documents`
            );
            
            this.spinner.classList.add('hidden');
            
            if (!response.ok) throw new Error('❌ Failed to load files');
            
            const data = await response.json();
            const files = data.documents || [];
            
            console.log(`📁 Loaded ${files.length} documents`);
            this.displayUploadedFiles(files);
            
        } catch (error) {
            this.spinner.classList.add('hidden');
            console.error('❌ Load files error:', error);
        }
    }

    displayUploadedFiles(files) {
        if (!this.uploadedFiles) return;
        
        if (files.length === 0) {
            this.uploadedFiles.innerHTML = '<p>⚠️ No files uploaded</p>';
            return;
        }
        
        this.uploadedFiles.innerHTML = files.map(file => {
            const isFromCurrentSession = this.uploadedFilesSession.some(f =>
                f.name === file.filename
            );
            const sessionBadge = isFromCurrentSession ? '✓ Current' : 'Previous';
            
            return `
                <div class="file-item">
                    <div class="file-details">
                        <span class="file-name">${file.filename || 'Unknown'}</span>
                        <span class="file-info">${file.chunk_count || 0} chunks</span>
                        <span class="file-badge">${sessionBadge}</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    async sendMessage() {
        if (!this.sessionId) {
            this.addMessage('system', '❌ Session not initialized', true);
            return;
        }
        
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.autoResizeTextarea();
        
        this.spinner.classList.remove('hidden');
        
        try {
            console.log(`💬 Sending message in session ${this.sessionId}...`);
            
            const response = await fetch(
                `${this.apiUrl}/api/session/${this.sessionId}/chat`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        use_rag: true,
                        n_chunks: 5
                    })
                }
            );
            
            this.spinner.classList.add('hidden');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: Request failed`);
            }
            
            const data = await response.json();
            
            console.log('✅ Response received:', data);
            
            // Separate reasoning and answer
            const { reasoning, answer } = this.separateReasoningAndAnswer(
                data.response
            );
            
            if (reasoning) {
                this.addMessage('reasoning', reasoning);
            }
            
            this.addMessage('assistant', answer);
            
            // Show S-GAS metadata
            if (data.metadata) {
                const meta = data.metadata;
                const metaText = `
📊 S-GAS Stats:
  • Iteration: ${meta.iteration || '?'}
  • Chunks used: ${meta.context_chunks_used || '?'}
  • New chunks: ${meta.new_chunks_in_this_iteration || '?'}
  • Total explored: ${meta.total_chunks_explored || '?'}
  • Coverage: ${meta.coverage_percent?.toFixed(1) || '?'}%
                `.trim();
                
                this.addMessage('system', metaText, true);
            }
            
        } catch (error) {
            this.spinner.classList.add('hidden');
            console.error('Send error:', error);
            this.addMessage('system', `❌ Error: ${error.message}`, true);
        }
    }

    async clearChat() {
        if (!this.sessionId) {
            this.addMessage('system', '❌ No session to clear', true);
            return;
        }
        
        if (!confirm('Are you sure? This will clear all chat and delete session data.')) {
            return;
        }
        
        try {
            this.spinner.classList.remove('hidden');
            
            console.log(`🗑️  Clearing session ${this.sessionId}...`);
            
            // Delete request to clear session
            const response = await fetch(
                `${this.apiUrl}/api/session/${this.sessionId}/clear`,
                { method: 'DELETE' }
            );
            
            this.spinner.classList.add('hidden');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: Clear failed`);
            }
            
            // Clear UI
            this.chatHistory = [];
            this.uploadedFilesSession = [];
            this.messageCount = 0;
            
            if (this.chatMessages) {
                this.chatMessages.innerHTML = '';
            }
            
            console.log('✅ Session cleared');
            
            // ✅ NEW: Reinitialize session
            await this.initializeSession();
            
            this.addMessage('system', '✅ Chat cleared and session reset', true);
            
        } catch (error) {
            this.spinner.classList.add('hidden');
            console.error('Clear error:', error);
            this.addMessage('system', `❌ Failed to clear: ${error.message}`, true);
        }
    }

    exportChat() {
        const chatData = {
            sessionId: this.sessionId,
            timestamp: new Date().toISOString(),
            messages: this.chatHistory
        };
        
        const dataStr = JSON.stringify(chatData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        
        link.href = url;
        link.download = `sgas-chat-${Date.now()}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
    }

    addMessage(type, content, isSystem = false) {
        this.messageCount++;
        
        const message = {
            type,
            content,
            timestamp: new Date().toISOString()
        };
        
        this.chatHistory.push(message);
        
        if (!this.chatMessages) return;
        
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}-message ${isSystem ? 'system' : ''}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        messageElement.appendChild(messageContent);
        this.chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        if (this.messageCountElement) {
            this.messageCountElement.textContent = this.messageCount;
        }
    }

    addFileUploadMessage(filename, status, message) {
        const content = `📄 ${filename}\n${message}`;
        const type = status === 'success' ? 'file-upload' : 'system';
        this.addMessage(type, content, true);
    }

    async checkServerStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            if (this.statusDot) {
                this.statusDot.style.backgroundColor = 'rgb(34, 197, 94)';
            }
            if (this.statusText) {
                this.statusText.textContent = '🟢 Connected';
            }
            
            if (this.modelNameElement && data.model_name) {
                this.modelNameElement.textContent = data.model_name;
            }
            if (this.embeddingDimElement && data.embedding_dim) {
                this.embeddingDimElement.textContent = data.embedding_dim;
            }
            
        } catch (error) {
            console.warn('Server status check failed:', error);
            if (this.statusDot) {
                this.statusDot.style.backgroundColor = 'rgb(239, 68, 68)';
            }
            if (this.statusText) {
                this.statusText.textContent = '🔴 Disconnected';
            }
        }
    }

    autoResizeTextarea() {
        if (!this.messageInput) return;
        
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(
            this.messageInput.scrollHeight,
            200
        ) + 'px';
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new SGASWebClient();
    });
} else {
    new SGASWebClient();
}