class TreeNode {
    constructor(name, value, children = []) {
        this.name = name;
        this.value = value;
        this.children = children;
        this.x = 0;
        this.y = 0;
        this.width = 0;
        this.height = 0;
        this.selected = false;
        this.hovered = false;
        this.level = 0;
    }
}

class TreeVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.tree = null;
        this.selectedNode = null;
        this.hoveredNode = null;
        
        // View settings
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.minScale = 0.1;
        this.maxScale = 3;
        
        // Layout settings
        this.nodeWidth = 220;
        this.nodeHeight = 120;
        this.levelWidth = 450; // Horizontal distance between levels
        this.siblingSpacing = 150; // Vertical spacing between siblings
        
        // Colors
        this.colors = {
            node: '#4A90E2',
            nodeHover: '#357ABD',
            nodeSelected: '#E74C3C',
            text: '#FFFFFF',
            line: '#666666',
            background: '#FAFAFA',
            paperNode: '#8E44AD',
            dataNode: '#27AE60',
            experimentNode: '#E67E22',
            metadataNode: '#95A5A6'
        };
        
        this.setupCanvas();
        this.setupEventListeners();
    }
    
    setupCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = Math.max(1600, container.clientWidth - 40);
        this.canvas.height = Math.max(900, window.innerHeight * 0.8);
        
        // If we have a tree loaded, recenter it
        if (this.tree) {
            this.centerTree();
        }
        this.draw();
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));
        
        // Prevent context menu
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Double click to expand/collapse
        this.canvas.addEventListener('dblclick', (e) => this.onDoubleClick(e));
    }
    
    loadTreeData(data) {
        this.tree = this.parseTreeData(data);
        this.layoutTree();
        this.centerTree();
        this.draw();
    }
    
    parseTreeData(data) {
        const node = new TreeNode(data.name, data.value);
        if (data.children) {
            node.children = data.children.map(child => this.parseTreeData(child));
        }
        return node;
    }
    
    layoutTree() {
        if (!this.tree) return;
        
        this.calculateLevels(this.tree, 0);
        
        // Position root node at origin
        this.tree.x = 0;
        this.tree.y = 0;
        
        this.calculatePositions(this.tree);
        
        // Adjust all positions to ensure no negative coordinates
        this.normalizePositions(this.tree);
    }
    
    normalizePositions(node) {
        // Find the minimum Y coordinate
        let minY = Infinity;
        const findMinY = (n) => {
            minY = Math.min(minY, n.y - n.height / 2);
            n.children.forEach(findMinY);
        };
        findMinY(node);
        
        // Adjust all positions to start from a reasonable Y position
        if (minY < 50) {
            const offset = 50 - minY;
            const adjustY = (n) => {
                n.y += offset;
                n.children.forEach(adjustY);
            };
            adjustY(node);
        }
    }
    
    calculateLevels(node, level) {
        node.level = level;
        node.children.forEach(child => this.calculateLevels(child, level + 1));
    }
    
    calculatePositions(node) {
        // Set node dimensions
        node.width = this.nodeWidth;
        node.height = this.nodeHeight;
        
        if (node.children.length === 0) {
            return;
        }
        
        // Calculate positions for children first
        node.children.forEach(child => this.calculatePositions(child));
        
        // Calculate total height needed for children (vertical spacing)
        const childrenHeight = node.children.reduce((sum, child) => sum + child.height, 0) + 
                              (node.children.length - 1) * this.siblingSpacing;
        
        // Position children to the right of parent, vertically centered
        let currentY = node.y - childrenHeight / 2;
        node.children.forEach(child => {
            child.x = node.x + this.levelWidth; // Fixed distance to the right
            child.y = currentY + child.height / 2;
            currentY += child.height + this.siblingSpacing;
        });
    }
    
    centerTree() {
        if (!this.tree) return;
        
        const treeBounds = this.getTreeBounds();
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        // Add padding around the tree
        const padding = 150;
        
        // Calculate the scale needed to fit the tree with padding
        const scaleX = (canvasWidth - 2 * padding) / treeBounds.width;
        const scaleY = (canvasHeight - 2 * padding) / treeBounds.height;
        const fitScale = Math.min(scaleX, scaleY, 1); // Don't scale up beyond 1
        
        // Set the scale to fit the tree
        this.scale = Math.max(this.minScale, fitScale);
        
        // Center the tree
        this.offsetX = canvasWidth / 2 - treeBounds.centerX * this.scale;
        this.offsetY = canvasHeight / 2 - treeBounds.centerY * this.scale;
        
        // Ensure the tree doesn't go off-screen
        const scaledWidth = treeBounds.width * this.scale;
        const scaledHeight = treeBounds.height * this.scale;
        
        if (scaledWidth > canvasWidth - 2 * padding) {
            this.offsetX = padding;
        }
        if (scaledHeight > canvasHeight - 2 * padding) {
            this.offsetY = padding;
        }
    }
    
    getTreeBounds() {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        
        const traverse = (node) => {
            const x = node.x;
            const y = node.y;
            
            minX = Math.min(minX, x - node.width / 2);
            maxX = Math.max(maxX, x + node.width / 2);
            minY = Math.min(minY, y - node.height / 2);
            maxY = Math.max(maxY, y + node.height / 2);
            
            node.children.forEach(traverse);
        };
        
        traverse(this.tree);
        
        return {
            minX, maxX, minY, maxY,
            centerX: (minX + maxX) / 2,
            centerY: (minY + maxY) / 2,
            width: maxX - minX,
            height: maxY - minY
        };
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.tree) return;
        
        this.ctx.save();
        this.ctx.scale(this.scale, this.scale);
        this.ctx.translate(this.offsetX, this.offsetY);
        
        this.drawTree(this.tree);
        
        this.ctx.restore();
    }
    
    drawTree(node) {
        // Draw connections to children first (so they appear behind nodes)
        this.drawConnections(node);
        
        // Draw the node
        this.drawNode(node);
        
        // Draw children recursively
        node.children.forEach(child => this.drawTree(child));
    }
    
    drawConnections(node) {
        if (node.children.length === 0) return;
        
        this.ctx.strokeStyle = this.colors.line;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        const startX = node.x;
        const startY = node.y + node.height / 2;
        
        node.children.forEach(child => {
            const endX = child.x;
            const endY = child.y - child.height / 2;
            
            this.ctx.moveTo(startX, startY);
            this.ctx.lineTo(startX, startY + (endY - startY) / 2);
            this.ctx.lineTo(endX, startY + (endY - startY) / 2);
            this.ctx.lineTo(endX, endY);
        });
        
        this.ctx.stroke();
    }
    
    drawNode(node) {
        const x = node.x - node.width / 2;
        const y = node.y - node.height / 2;
        
        // Determine node color based on type
        let nodeColor = this.getNodeColor(node);
        
        if (node.selected) {
            nodeColor = this.colors.nodeSelected;
        } else if (node.hovered) {
            nodeColor = this.lightenColor(nodeColor, 20);
        }
        
        // Draw node background with rounded corners
        this.drawRoundedRect(x, y, node.width, node.height, 8);
        this.ctx.fillStyle = nodeColor;
        this.ctx.fill();
        
        // Draw node border
        this.ctx.strokeStyle = '#333';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw text
        this.ctx.fillStyle = this.colors.text;
        this.ctx.font = 'bold 13px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        // Draw node name
        this.ctx.fillText(this.truncateText(node.name, 18), node.x, node.y - 10);
        
        // Draw node value (if it's a leaf or has a meaningful value)
        if (node.children.length === 0 || node.value !== node.name) {
            this.ctx.font = '11px Arial';
            this.ctx.fillText(this.truncateText(node.value, 25), node.x, node.y + 10);
        }
    }
    
    drawRoundedRect(x, y, width, height, radius) {
        this.ctx.beginPath();
        this.ctx.moveTo(x + radius, y);
        this.ctx.lineTo(x + width - radius, y);
        this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        this.ctx.lineTo(x + width, y + height - radius);
        this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        this.ctx.lineTo(x + radius, y + height);
        this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        this.ctx.lineTo(x, y + radius);
        this.ctx.quadraticCurveTo(x, y, x + radius, y);
        this.ctx.closePath();
    }
    
    getNodeColor(node) {
        const name = node.name.toLowerCase();
        
        if (name === 'paper') return this.colors.paperNode;
        if (name.includes('data') || name.includes('sample')) return this.colors.dataNode;
        if (name.includes('experiment') || name.includes('metric')) return this.colors.experimentNode;
        if (name.includes('metadata') || name.includes('version') || name.includes('id')) return this.colors.metadataNode;
        
        return this.colors.node;
    }
    
    lightenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }
    
    // Event handlers
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale - this.offsetX;
        const y = (e.clientY - rect.top) / this.scale - this.offsetY;
        
        const clickedNode = this.getNodeAt(x, y);
        if (clickedNode) {
            this.selectNode(clickedNode);
        }
        
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
    }
    
    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale - this.offsetX;
        const y = (e.clientY - rect.top) / this.scale - this.offsetY;
        
        // Update hover state
        const hoveredNode = this.getNodeAt(x, y);
        if (hoveredNode !== this.hoveredNode) {
            if (this.hoveredNode) this.hoveredNode.hovered = false;
            this.hoveredNode = hoveredNode;
            if (hoveredNode) hoveredNode.hovered = true;
            this.draw();
        }
        
        // Handle dragging
        if (this.isDragging) {
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            
            this.offsetX += deltaX / this.scale;
            this.offsetY += deltaY / this.scale;
            
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            
            this.draw();
        }
    }
    
    onMouseUp(e) {
        this.isDragging = false;
    }
    
    onWheel(e) {
        e.preventDefault();
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * zoomFactor));
        
        if (newScale !== this.scale) {
            const scaleChange = newScale / this.scale;
            
            // Zoom towards mouse position
            this.offsetX = mouseX - (mouseX - this.offsetX) * scaleChange;
            this.offsetY = mouseY - (mouseY - this.offsetY) * scaleChange;
            
            this.scale = newScale;
            this.draw();
        }
    }
    
    onDoubleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / this.scale - this.offsetX;
        const y = (e.clientY - rect.top) / this.scale - this.offsetY;
        
        const clickedNode = this.getNodeAt(x, y);
        if (clickedNode) {
            // Toggle node expansion (for future use)
            console.log('Double-clicked node:', clickedNode.name);
        }
    }
    
    getNodeAt(x, y) {
        if (!this.tree) return null;
        
        const traverse = (node) => {
            const nodeX = node.x;
            const nodeY = node.y;
            const halfWidth = node.width / 2;
            const halfHeight = node.height / 2;
            
            if (x >= nodeX - halfWidth && x <= nodeX + halfWidth &&
                y >= nodeY - halfHeight && y <= nodeY + halfHeight) {
                return node;
            }
            
            for (const child of node.children) {
                const result = traverse(child);
                if (result) return result;
            }
            
            return null;
        };
        
        return traverse(this.tree);
    }
    
    selectNode(node) {
        if (this.selectedNode) {
            this.selectedNode.selected = false;
        }
        
        this.selectedNode = node;
        node.selected = true;
        this.draw();
        
        // Dispatch custom event for annotation tool
        const event = new CustomEvent('nodeSelected', {
            detail: { node: node }
        });
        document.dispatchEvent(event);
    }
    
    resetView() {
        this.scale = 1;
        this.centerTree();
        this.draw();
    }
    
    // Utility methods for external access
    getSelectedNode() {
        return this.selectedNode;
    }
    
    clearSelection() {
        if (this.selectedNode) {
            this.selectedNode.selected = false;
            this.selectedNode = null;
            this.draw();
        }
    }
}
