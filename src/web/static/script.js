class SGASWebClient {
    constructor() {
        this.apiUrl = 'http://127.0.0.1:8080';
        this.messageCount = 0;
        this.chatHistory = [];
        
        this.initializeElements();
        this.bindEvents();
        this.checkServerStatus();
        
        console.log('S-GAS Web Client initialized');
    }

    initializeElements() {
        // Main elements
        this.messageForm = document.getElementById('messageForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.useRagCheckbox = document.getElementById('useRag');
        
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
    }

    bindEvents() {
        // Sending message
        this.messageForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Sending message with Ctrl+Enter
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Clear the chat
        this.clearChatButton.addEventListener('click', () => {
            this.clearChat();
        });
        
        // Export the chat
        this.exportChatButton.addEventListener('click', () => {
            this.exportChat();
        });

        // Auto changing the size of text block
        this.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });
    }

    autoResizeTextarea() {
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
                
                if (data.model) {
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
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = message;
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        const useRag = this.useRagCheckbox.checked;

        // Turning off the UI
        this.setLoading(true);

        // Adding a user message to the chat
        this.addMessage('user', message);
        
        // Clearing the input field
        this.messageInput.value = '';
        this.autoResizeTextarea();

        // Adding typing animation
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

            // Removing the typing indicator
            this.removeTypingIndicator(typingId);

            if (response.ok) {
                const data = await response.json();
                
                // Adding a reply to the chat
                this.addMessage('assistant', data.response);
                
                // Updating metadata
                this.updateMetadata(data.metadata);
                
                // Updating information about the last request
                this.updateRequestInfo(message, data);
                
            } else {
                const errorData = await response.json().catch(() => ({}));
                this.addMessage('system', `Error: ${errorData.detail || 'Unknown server error'}`);
            }

        } catch (error) {
            this.removeTypingIndicator(typingId);
            this.addMessage('system', `Connection error: ${error.message}`);
            console.error('Error sending message:', error);
        }

        // Turning on the UI
        this.setLoading(false);
    }

    addMessage(sender, content, isSystem = false) {
        // Delete the system message when got the first query by user
        if (sender === 'user' && this.messageCount === 0) {
            this.removeWelcomeMessage();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isSystem ? 'system-message' : sender}`;
        
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-EN', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        messageDiv.innerHTML = `
            <div class="message-content">${this.formatMessage(content)}</div>
            <div class="message-time">${timeString}</div>
        `;

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        // Updating the message counter and history
        if (!isSystem) {
            if (sender === 'user') {
                this.messageCount++;
                this.messageCountElement.textContent = this.messageCount;
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
        const welcomeMessage = this.chatMessages.querySelector('.system-message');
        if (welcomeMessage) {
            //welcomeMessage.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => {
                if (welcomeMessage.parentNode) {
                    welcomeMessage.remove();
                }
            }, 300);
        }
    }

    formatMessage(content) {
        // Basic text formatting
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        
        const typingId = Date.now();
        typingDiv.id = `typing-${typingId}`;
        
        typingDiv.innerHTML = `
            <div class="message-content">
                <span class="typing-animation">Is typing...</span>
            </div>
        `;

        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
        
        return typingId;
    }

    removeTypingIndicator(typingId) {
        const typingElement = document.getElementById(`typing-${typingId}`);
        if (typingElement) {
            typingElement.remove();
        }
    }

    updateMetadata(metadata) {
        if (metadata.embedding_shape) {
            // Extract the dimension from the array [1, 384] -> 384
            const dimension = Array.isArray(metadata.embedding_shape) 
                ? metadata.embedding_shape[1] || metadata.embedding_shape[0]
                : metadata.embedding_shape;
            this.embeddingDimElement.textContent = dimension;
        }
        
        if (metadata.model_used) {
            this.modelNameElement.textContent = metadata.model_used;
        }
    }

    updateRequestInfo(query, response) {
        const info = `
            <strong>Query:</strong> ${query.length > 50 ? query.substring(0, 50) + '...' : query}<br>
            <strong>Answer:</strong> ${response.response.length} symbols<br>
            <strong>RAG:</strong> ${this.useRagCheckbox.checked ? 'is turned on' : 'is turned off'}<br>
            <strong>Time:</strong> ${new Date().toLocaleTimeString('en-EN')}
        `;
        this.requestInfoElement.innerHTML = info;
    }

    setLoading(isLoading) {
        this.sendButton.disabled = isLoading;
        this.messageInput.disabled = isLoading;
        
        if (isLoading) {
            this.sendButton.classList.add('loading');
        } else {
            this.sendButton.classList.remove('loading');
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    clearChat() {
        // Deleting all messages
        this.chatMessages.innerHTML = '';

        // Restoring the welcome message
        const welcomeHtml = `
            <div class="message system-message">
                <div class="message-content">
                    <strong>System is ready</strong><br>
                    Enter your message...
                </div>
            </div>
        `;
        this.chatMessages.insertAdjacentHTML('beforeend', welcomeHtml);

        // Reseting the counters
        this.messageCount = 0;
        this.messageCountElement.textContent = '0';
        this.embeddingDimElement.textContent = '-';
        this.requestInfoElement.innerHTML = '<p class="no-data">No data</p>';

        // Clearing the chat
        this.chatHistory = [];

        console.log('Chat cleared and welcome message restored');
    }


    exportChat() {
        if (this.chatHistory.length === 0) {
            alert('The history of chat is empty');
            return;
        }

        const exportData = {
            export_date: new Date().toISOString(),
            message_count: this.messageCount,
            model: this.modelNameElement.textContent,
            embedding_dimension: this.embeddingDimElement.textContent,
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
        
        console.log('Chat was exported');
    }
}

// Initialization on page load
document.addEventListener('DOMContentLoaded', () => {
    window.sgasClient = new SGASWebClient();
});

// Global error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

// Status update every 30 seconds
setInterval(() => {
    if (window.sgasClient) {
        window.sgasClient.checkServerStatus();
    }
}, 30000);