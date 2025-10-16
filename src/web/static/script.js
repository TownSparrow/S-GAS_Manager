class SGASWebClient {
    constructor() {
        this.apiUrl = 'http://127.0.0.1:8080';
        this.messageCount = 0;
        this.chatHistory = [];
        this.uploadedFilesSession = [];
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    initialize() {
        console.log('Initializing S-GAS Web Client...');
        this.initializeElements();
        this.bindEvents();
        this.checkServerStatus();
        this.loadUploadedFiles();
        console.log('S-GAS Web Client initialized successfully');
    }

    initializeElements() {
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

        console.log('Elements initialized');
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

        console.log('Events bound successfully');
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

        const allowedTypes = ['.pdf', '.txt', '.docx', '.doc'];
        const validFiles = Array.from(files).filter(file => {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            return allowedTypes.includes(extension);
        });

        if (validFiles.length === 0) {
            this.addMessage('system', 'Supported file types: PDF, TXT, DOCX, DOC', true);
            return;
        }

        this.displaySelectedFiles(validFiles);

        for (const file of validFiles) {
            try {
                await this.uploadFile(file);
            } catch (error) {
                console.error('Error uploading file:', error);
                this.addFileUploadMessage(file.name, 'error', error.message);
            }
        }

        // Clear the file input after upload
        this.clearFileInput();
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

    displaySelectedFiles(files) {
        if (!this.fileList) return;
        
        this.fileList.innerHTML = '';
        this.fileList.classList.remove('hidden');

        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div class="file-info">
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${this.formatFileSize(file.size)}</span>
                </div>
                <div class="file-status">
                    <div class="upload-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            `;
            this.fileList.appendChild(fileItem);
        });
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const fileItem = Array.from(this.fileList.children).find(item => 
            item.querySelector('.file-name').textContent === file.name
        );

        try {
            const response = await fetch(`${this.apiUrl}/api/upload-document`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                
                if (fileItem) {
                    const progressFill = fileItem.querySelector('.progress-fill');
                    progressFill.style.width = '100%';
                    progressFill.style.backgroundColor = 'var(--color-success)';
                    
                    const fileStatus = fileItem.querySelector('.file-status');
                    fileStatus.innerHTML = '<span class="status status--success">✓ Uploaded</span>';
                }

                this.uploadedFilesSession.push({
                    name: file.name,
                    size: file.size,
                    uploadTime: new Date()
                });
                
                // Add file upload message with delete button
                this.addFileUploadMessage(file.name, 'success');
                
                setTimeout(() => this.loadUploadedFiles(), 500);

            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            if (fileItem) {
                const fileStatus = fileItem.querySelector('.file-status');
                fileStatus.innerHTML = '<span class="status status--error">✗ Error</span>';
            }
            throw error;
        }
    }

    addFileUploadMessage(filename, status, errorMessage = null) {
        if (!this.chatMessages) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message file-upload-message';
        messageDiv.dataset.filename = filename;

        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });

        let content = '';
        if (status === 'success') {
            content = `
                <div class="file-upload-content">
                    <div class="file-upload-info">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                        <span class="file-upload-name">${filename}</span>
                        <span class="file-upload-status success">✓ Uploaded</span>
                    </div>
                    <button class="btn btn--sm btn--outline delete-uploaded-file" data-filename="${filename}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                        Delete
                    </button>
                </div>
            `;
        } else {
            content = `
                <div class="file-upload-content">
                    <div class="file-upload-info">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                        <span class="file-upload-name">${filename}</span>
                        <span class="file-upload-status error">✗ Error</span>
                    </div>
                    <p class="error-message">${errorMessage || 'Upload failed'}</p>
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="message-content">
                ${content}
            </div>
            <div class="message-time">
                ${timeString}
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        // Bind delete button event
        if (status === 'success') {
            const deleteBtn = messageDiv.querySelector('.delete-uploaded-file');
            if (deleteBtn) {
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.deleteFileFromMessage(filename, messageDiv);
                });
            }
        }
    }

    async deleteFileFromMessage(filename, messageElement) {
        if (!confirm(`Delete file "${filename}" from server?`)) return;

        try {
            const response = await fetch(`${this.apiUrl}/api/documents/${filename}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                // Remove from session files
                this.uploadedFilesSession = this.uploadedFilesSession.filter(f => f.name !== filename);
                
                // Remove message from chat
                if (messageElement && messageElement.parentNode) {
                    messageElement.remove();
                }
                
                // Add system message
                this.addMessage('system', `File "${filename}" deleted successfully`, true);
                
                // Refresh file list
                this.loadUploadedFiles();
            } else {
                this.addMessage('system', `Error deleting file "${filename}"`, true);
            }
        } catch (error) {
            console.error('Error deleting file:', error);
            this.addMessage('system', `Error deleting file "${filename}": ${error.message}`, true);
        }
    }

    async loadUploadedFiles() {
        try {
            const response = await fetch(`${this.apiUrl}/api/documents`);
            if (response.ok) {
                const data = await response.json();
                this.displayUploadedFiles(data.documents || []);
            }
        } catch (error) {
            console.error('Error loading uploaded files:', error);
        }
    }

    displayUploadedFiles(files) {
        if (!this.uploadedFiles) return;
        
        if (!files || files.length === 0) {
            this.uploadedFiles.innerHTML = '<p class="no-files-text">No files uploaded</p>';
            return;
        }

        this.uploadedFiles.innerHTML = files.map(file => {
            const isFromCurrentSession = this.uploadedFilesSession.some(f => f.name === file.filename);
            const sessionBadge = isFromCurrentSession ? '<span class="session-badge">Current Session</span>' : '';
            
            return `
                <div class="uploaded-file-item ${isFromCurrentSession ? 'session-file' : ''}">
                    <div class="file-header">
                        <span class="file-name">${file.filename}</span>
                        ${sessionBadge}
                    </div>
                    <div class="file-details">
                        <span class="file-size">${this.formatFileSize(file.size)}</span>
                        <button class="btn btn--outline btn--sm delete-file" data-filename="${file.filename}">
                            Delete
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        this.uploadedFiles.querySelectorAll('.delete-file').forEach(button => {
            button.addEventListener('click', (e) => {
                this.deleteFile(e.target.dataset.filename);
            });
        });
    }

    async deleteFile(filename) {
        if (!confirm(`Delete file "${filename}"?`)) return;

        try {
            const response = await fetch(`${this.apiUrl}/api/documents/${filename}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.addMessage('system', `File "${filename}" deleted`, true);
                
                this.uploadedFilesSession = this.uploadedFilesSession.filter(f => f.name !== filename);
                
                // Remove file upload message from chat
                const fileMessages = this.chatMessages.querySelectorAll(`.file-upload-message[data-filename="${filename}"]`);
                fileMessages.forEach(msg => msg.remove());
                
                this.loadUploadedFiles();
            } else {
                this.addMessage('system', `Error deleting file "${filename}"`, true);
            }
        } catch (error) {
            console.error('Error deleting file:', error);
            this.addMessage('system', `Error deleting file "${filename}": ${error.message}`, true);
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    autoResizeTextarea() {
        if (!this.messageInput) return;
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
    }

    async checkServerStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.updateStatus('online', `Server online • vLLM: ${data.vllm_status}`);
                if (data.model && this.modelNameElement) {
                    this.modelNameElement.textContent = data.model;
                }
            } else {
                this.updateStatus('error', 'Server error');
            }
        } catch (error) {
            this.updateStatus('error', 'Server unavailable');
            console.error('Error checking server status:', error);
        }
    }

    updateStatus(status, message) {
        if (this.statusDot) {
            this.statusDot.className = `status-dot ${status}`;
        }
        if (this.statusText) {
            this.statusText.textContent = message;
        }
    }

    async sendMessage() {
        if (!this.messageInput) return;
        
        const message = this.messageInput.value.trim();
        if (!message) return;

        const useRag = this.uploadedFilesSession.length > 0;

        this.setLoading(true);
        this.addMessage('user', message);

        this.messageInput.value = '';
        this.autoResizeTextarea();

        const typingId = this.addTypingIndicator();

        try {
            const response = await fetch(`${this.apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    use_rag: useRag
                })
            });

            this.removeTypingIndicator(typingId);

            if (response.ok) {
                const data = await response.json();
                
                const { reasoning, answer } = this.separateReasoningAndAnswer(data.response);
                
                if (reasoning) {
                    this.addMessage('reasoning', reasoning, false);
                }
                
                this.addMessage('assistant', answer);
                
                this.updateMetadata(data.metadata);
                this.updateRequestInfo(message, data);
            } else {
                const errorData = await response.json().catch(() => ({}));
                this.addMessage('system', `Error: ${errorData.detail || 'Unknown server error'}`, true);
            }
        } catch (error) {
            this.removeTypingIndicator(typingId);
            this.addMessage('system', `Connection error: ${error.message}`, true);
            console.error('Error sending message:', error);
        }

        this.setLoading(false);
    }

    addMessage(sender, content, isSystem = false) {
        if (!this.chatMessages) return;
        
        if (sender === 'user' && this.messageCount === 0) {
            this.removeWelcomeMessage();
        }

        const messageDiv = document.createElement('div');
        
        if (sender === 'reasoning') {
            messageDiv.className = 'message reasoning-message';
        } else if (isSystem) {
            messageDiv.className = 'message system-message';
        } else {
            messageDiv.className = `message ${sender}`;
        }

        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });

        messageDiv.innerHTML = `
            <div class="message-content">
                ${this.formatMessage(content)}
            </div>
            <div class="message-time">
                ${timeString}
            </div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        if (!isSystem && sender !== 'reasoning') {
            if (sender === 'user') {
                this.messageCount++;
                if (this.messageCountElement) {
                    this.messageCountElement.textContent = this.messageCount;
                }
            }
            
            this.chatHistory.push({
                sender,
                content,
                timestamp: now.toISOString(),
                formatted_time: timeString
            });
        }
    }

    removeWelcomeMessage() {
        if (!this.chatMessages) return;
        const welcomeMessage = this.chatMessages.querySelector('.system-message');
        if (welcomeMessage) {
            setTimeout(() => {
                if (welcomeMessage.parentNode) {
                    welcomeMessage.remove();
                }
            }, 300);
        }
    }

    formatMessage(content) {
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    addTypingIndicator() {
        if (!this.chatMessages) return null;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        const typingId = Date.now();
        typingDiv.id = `typing-${typingId}`;
        
        typingDiv.innerHTML = `
            <div class="message-content typing-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
        return typingId;
    }

    removeTypingIndicator(typingId) {
        if (!typingId) return;
        const typingElement = document.getElementById(`typing-${typingId}`);
        if (typingElement) {
            typingElement.remove();
        }
    }

    updateMetadata(metadata) {
        if (metadata.embedding_shape && this.embeddingDimElement) {
            const dimension = Array.isArray(metadata.embedding_shape)
                ? metadata.embedding_shape[1] || metadata.embedding_shape[0]
                : metadata.embedding_shape;
            this.embeddingDimElement.textContent = dimension;
        }
        
        if (metadata.model_used && this.modelNameElement) {
            this.modelNameElement.textContent = metadata.model_used;
        }
    }

    updateRequestInfo(query, response) {
        if (!this.requestInfoElement) return;
        
        const info = `
            <div class="request-detail">
                <strong>Query:</strong> ${query.substring(0, 50)}${query.length > 50 ? '...' : ''}
            </div>
            <div class="request-detail">
                <strong>RAG:</strong> ${response.metadata.use_rag ? 'Enabled' : 'Disabled'}
            </div>
            <div class="request-detail">
                <strong>Chunks:</strong> ${response.metadata.context_chunks_used || 0}
            </div>
            <div class="request-detail">
                <strong>Time:</strong> ${new Date(response.metadata.timestamp).toLocaleTimeString()}
            </div>
        `;
        
        this.requestInfoElement.innerHTML = info;
    }

    setLoading(loading) {
        if (this.sendButton) {
            this.sendButton.disabled = loading;
            this.sendButton.textContent = loading ? 'Sending...' : 'Send';
        }
        if (this.uploadButton) {
            this.uploadButton.disabled = loading;
        }
        if (this.spinner) {
            if (loading) {
                this.spinner.classList.remove('hidden');
            } else {
                this.spinner.classList.add('hidden');
            }
        }
    }

    scrollToBottom() {
        if (this.chatMessages) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }

    clearChat() {
        if (!this.chatMessages) return;
        
        this.chatMessages.innerHTML = `
            <div class="message system-message">
                <div class="message-content">
                    Welcome to S-GAS Manager! Send a message to start the conversation.
                </div>
            </div>
        `;
        
        this.messageCount = 0;
        if (this.messageCountElement) {
            this.messageCountElement.textContent = '0';
        }
        if (this.requestInfoElement) {
            this.requestInfoElement.innerHTML = 'No data';
        }
        
        this.uploadedFilesSession = [];
        this.chatHistory = [];
        
        console.log('Chat cleared');
    }

    exportChat() {
        if (this.chatHistory.length === 0) {
            alert('Chat history is empty');
            return;
        }

        const exportData = {
            export_date: new Date().toISOString(),
            message_count: this.messageCount,
            model: this.modelNameElement ? this.modelNameElement.textContent : '',
            embedding_dimension: this.embeddingDimElement ? this.embeddingDimElement.textContent : '',
            uploaded_files: this.uploadedFilesSession,
            chat_history: this.chatHistory
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});

        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `sgas-chat-export-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        console.log('Chat exported');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.sgasClient = new SGASWebClient();
});

window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

setInterval(() => {
    if (window.sgasClient) {
        window.sgasClient.checkServerStatus();
    }
}, 30000);