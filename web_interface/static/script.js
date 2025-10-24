// 全局变量
let cy; // Cytoscape 实例
let objectTypes = {};
let relationTypes = {};
let currentRoomType = 'bedroom';
let nodeIdCounter = 0;
let templates = {};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initCytoscape();
    loadObjectTypes();
    loadRelationTypes();
    loadTemplates();
    
    // 初始化3D场景
    if (typeof init3DScene === 'function') {
        init3DScene();
        console.log('3D场景初始化完成');
    } else {
        console.warn('viewer3d.js未加载');
    }
    
    // CFG Scale 滑块事件
    document.getElementById('cfgScale').addEventListener('input', function(e) {
        document.getElementById('cfgScaleValue').textContent = e.target.value;
    });
});

// 初始化 Cytoscape 场景图编辑器
function initCytoscape() {
    cy = cytoscape({
        container: document.getElementById('cy'),
        
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#667eea',
                    'label': 'data(label)',
                    'color': '#fff',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '14px',
                    'font-weight': 'bold',
                    'width': '80px',
                    'height': '80px',
                    'text-wrap': 'wrap',
                    'text-max-width': '70px',
                    'border-width': 3,
                    'border-color': '#5568d3'
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'background-color': '#f6ad55',
                    'border-color': '#ed8936',
                    'border-width': 4
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 3,
                    'line-color': '#48bb78',
                    'target-arrow-color': '#48bb78',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '12px',
                    'color': '#333',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -10,
                    'text-background-color': '#fff',
                    'text-background-opacity': 0.8,
                    'text-background-padding': '3px'
                }
            },
            {
                selector: 'edge:selected',
                style: {
                    'line-color': '#f6ad55',
                    'target-arrow-color': '#f6ad55',
                    'width': 4
                }
            }
        ],
        
        layout: {
            name: 'circle',
            padding: 50
        }
    });
    
    // 右键删除节点/边
    cy.on('cxttap', 'node', function(evt) {
        if (confirm('确定要删除这个物体吗？')) {
            cy.remove(evt.target);
            updateStats();
            updateRelationSelects();
        }
    });
    
    cy.on('cxttap', 'edge', function(evt) {
        if (confirm('确定要删除这个关系吗？')) {
            cy.remove(evt.target);
            updateStats();
        }
    });
    
    // 选中节点时高亮
    cy.on('select', 'node', function(evt) {
        console.log('选中物体:', evt.target.data('label'));
    });
    
    updateStats();
}

// 加载物体类型
async function loadObjectTypes() {
    try {
        const response = await fetch(`/api/object_types/${currentRoomType}`);
        const data = await response.json();
        objectTypes = {};
        
        const select = document.getElementById('objectTypeSelect');
        select.innerHTML = '';
        
        data.object_types.forEach(obj => {
            objectTypes[obj.id] = obj.name;
            const option = document.createElement('option');
            option.value = obj.id;
            option.textContent = `${obj.name} (${obj.id})`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('加载物体类型失败:', error);
    }
}

// 加载关系类型
async function loadRelationTypes() {
    try {
        const response = await fetch('/api/relation_types');
        const data = await response.json();
        relationTypes = {};
        
        data.relation_types.forEach(rel => {
            relationTypes[rel.id] = rel.name;
        });
    } catch (error) {
        console.error('加载关系类型失败:', error);
    }
}

// 加载模板
async function loadTemplates() {
    try {
        const response = await fetch(`/api/templates/${currentRoomType}`);
        const data = await response.json();
        templates = data.templates || {};
        
        const select = document.getElementById('templateSelect');
        select.innerHTML = '<option value="">-- 选择模板 --</option>';
        
        Object.keys(templates).forEach(key => {
            const template = templates[key];
            const option = document.createElement('option');
            option.value = key;
            option.textContent = `${template.name} - ${template.description}`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('加载模板失败:', error);
    }
}

// 房间类型改变
function onRoomTypeChange() {
    currentRoomType = document.getElementById('roomType').value;
    loadObjectTypes();
    loadTemplates();
    clearScene();
}

// 加载模板
function loadTemplate() {
    const templateKey = document.getElementById('templateSelect').value;
    if (!templateKey) return;
    
    const template = templates[templateKey];
    if (!template) return;
    
    // 直接清空场景，不弹出确认框
    cy.elements().remove();
    nodeIdCounter = 0;
    
    // 清空右侧结果面板
    const viewData = document.getElementById('viewData');
    if (viewData) {
        viewData.innerHTML = `
            <div class="empty-state">
                <p>🎯 请配置场景图并点击"生成3D布局"</p>
            </div>
        `;
    }
    
    // 添加物体
    template.objects.forEach((objId, idx) => {
        const objName = objectTypes[objId] || `Object ${objId}`;
        cy.add({
            group: 'nodes',
            data: {
                id: `node${nodeIdCounter}`,
                label: objName,
                typeId: objId,
                nodeIndex: idx
            }
        });
        nodeIdCounter++;
    });
    
    // 添加关系
    if (template.edges) {
        template.edges.forEach(edge => {
            const sourceNode = cy.nodes().filter(n => n.data('nodeIndex') === edge.source)[0];
            const targetNode = cy.nodes().filter(n => n.data('nodeIndex') === edge.target)[0];
            
            if (sourceNode && targetNode) {
                const relName = relationTypes[edge.relation] || `Rel ${edge.relation}`;
                cy.add({
                    group: 'edges',
                    data: {
                        source: sourceNode.id(),
                        target: targetNode.id(),
                        label: relName,
                        relationId: edge.relation
                    }
                });
            }
        });
    }
    
    // 重新布局
    cy.layout({
        name: 'circle',
        padding: 50
    }).run();
    
    updateStats();
    updateRelationSelects();
    
    console.log(`已加载模板: ${template.name}`);
}

// 添加物体
function addObject() {
    const typeId = parseInt(document.getElementById('objectTypeSelect').value);
    const typeName = objectTypes[typeId];
    
    if (!typeName) {
        alert('请选择物体类型');
        return;
    }
    
    cy.add({
        group: 'nodes',
        data: {
            id: `node${nodeIdCounter}`,
            label: typeName,
            typeId: typeId,
            nodeIndex: cy.nodes().length
        }
    });
    
    nodeIdCounter++;
    
    // 重新布局
    cy.layout({
        name: 'circle',
        padding: 50
    }).run();
    
    updateStats();
    updateRelationSelects();
}

// 添加关系
function addRelation() {
    const sourceId = document.getElementById('relationSource').value;
    const targetId = document.getElementById('relationTarget').value;
    const relationId = parseInt(document.getElementById('relationType').value);
    
    if (!sourceId || !targetId) {
        alert('请选择源物体和目标物体');
        return;
    }
    
    if (sourceId === targetId) {
        alert('源物体和目标物体不能相同');
        return;
    }
    
    // 检查是否已存在相同的边
    const existingEdge = cy.edges().filter(e => 
        e.data('source') === sourceId && e.data('target') === targetId
    );
    
    if (existingEdge.length > 0) {
        alert('该关系已存在');
        return;
    }
    
    const relationName = relationTypes[relationId];
    
    cy.add({
        group: 'edges',
        data: {
            source: sourceId,
            target: targetId,
            label: relationName,
            relationId: relationId
        }
    });
    
    updateStats();
}

// 更新关系选择框
function updateRelationSelects() {
    const sourceSelect = document.getElementById('relationSource');
    const targetSelect = document.getElementById('relationTarget');
    
    sourceSelect.innerHTML = '<option value="">请选择</option>';
    targetSelect.innerHTML = '<option value="">请选择</option>';
    
    cy.nodes().forEach(node => {
        const option1 = document.createElement('option');
        option1.value = node.id();
        option1.textContent = node.data('label');
        sourceSelect.appendChild(option1);
        
        const option2 = document.createElement('option');
        option2.value = node.id();
        option2.textContent = node.data('label');
        targetSelect.appendChild(option2);
    });
}

// 更新统计信息
function updateStats() {
    const objectCount = document.getElementById('objectCount');
    const relationCount = document.getElementById('relationCount');
    
    if (objectCount) {
        objectCount.textContent = cy.nodes().length;
    }
    if (relationCount) {
        relationCount.textContent = cy.edges().length;
    }
}

// 重置布局
function resetLayout() {
    cy.layout({
        name: 'circle',
        padding: 50
    }).run();
}

// 导出场景图
function exportSceneGraph() {
    const sceneData = {
        room_type: currentRoomType,
        objects: cy.nodes().map(node => node.data('typeId')),
        edges: cy.edges().map(edge => ({
            source: cy.nodes().indexOf(cy.getElementById(edge.data('source'))),
            target: cy.nodes().indexOf(cy.getElementById(edge.data('target'))),
            relation: edge.data('relationId')
        }))
    };
    
    const dataStr = JSON.stringify(sceneData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `scene_graph_${currentRoomType}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

// 生成场景
async function generateScene() {
    if (cy.nodes().length === 0) {
        alert('请先添加物体');
        return;
    }
    
    // 清空生成日志
    clearProcessLog();
    
    // 构建场景数据
    const objects = cy.nodes().map(node => node.data('typeId'));
    const edges = [];
    
    addProcessLog('step', '构建场景图', `场景包含 ${cy.nodes().length} 个物体，${cy.edges().length} 个关系`);
    
    cy.edges().forEach(edge => {
        const sourceNode = cy.getElementById(edge.data('source'));
        const targetNode = cy.getElementById(edge.data('target'));
        const sourceIdx = cy.nodes().indexOf(sourceNode);
        const targetIdx = cy.nodes().indexOf(targetNode);
        
        edges.push({
            source: sourceIdx,
            target: targetIdx,
            relation: edge.data('relationId')
        });
    });
    
    const cfgScale = parseFloat(document.getElementById('cfgScale').value);
    const seedInput = document.getElementById('randomSeed').value;
    const seed = seedInput ? parseInt(seedInput) : null;
    
    addProcessLog('info', '配置参数', 
        `CFG Scale: ${cfgScale}\n随机种子: ${seed || '随机'}`,
        `room_type: ${currentRoomType}\ndevice: cuda:0`
    );
    
    const requestData = {
        room_type: currentRoomType,
        objects: objects,
        edges: edges,
        cfg_scale: cfgScale,
        seed: seed,
        device: 'cuda:0',
        scene_name: `web_${Date.now()}`,
        render_images: false  // 不需要渲染图片，只需要导出OBJ
    };
    
    // 显示加载状态
    const loadingOverlay = document.getElementById('loadingOverlay');
    loadingOverlay.style.display = 'flex';
    loadingOverlay.innerHTML = '<div class="spinner"></div><p>正在生成场景（AI推理中）...</p><p style="font-size: 12px; opacity: 0.7; margin-top: 10px;">这可能需要1-2分钟</p>';
    
    addProcessLog('step', '开始AI推理', '正在调用扩散模型生成场景布局...');
    
    try {
        const startTime = Date.now();
        
        const response = await fetch('/api/render_scene', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const error = await response.json();
            addProcessLog('error', '生成失败', error.error || '未知错误');
            throw new Error(error.error || '生成失败');
        }
        
        const result = await response.json();
        const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
        
        // 显示生成过程日志
        if (result.process_log) {
            result.process_log.forEach(log => {
                addProcessLog(log.type || 'info', log.title, log.content, log.details);
            });
        }
        
        addProcessLog('success', '场景生成完成', 
            `总耗时: ${elapsedTime}秒\n场景名称: ${result.scene_name}`,
            `导出目录: ${result.export_dir || '未知'}`
        );
        
        displayResult(result);
        
    } catch (error) {
        addProcessLog('error', '生成失败', error.message);
        alert('生成场景失败: ' + error.message);
        console.error(error);
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
}

// 显示生成结果
function displayResult(result) {
    const panel = document.getElementById('viewData');
    if (!panel) {
        console.error('viewData元素不存在');
        return;
    }
    
    panel.innerHTML = '';
    
    // 检查是否是Blender导出的场景（新格式）
    if (result.scene_url) {
        // 显示3D场景
        try {
            display3DScene(result);
            console.log('正在加载Blender场景...');
        } catch (error) {
            console.error('显示3D场景失败:', error);
        }
        
        // 添加成功提示
        const successDiv = document.createElement('div');
        successDiv.style.cssText = 'padding: 15px; background: #c6f6d5; color: #22543d; border-radius: 6px; margin-bottom: 15px;';
        successDiv.innerHTML = `
            <div style="text-align: center; font-size: 18px; margin-bottom: 10px;">✅ 场景已生成！</div>
            <div style="font-size: 12px;">场景名称: ${result.scene_name}</div>
            <div style="font-size: 12px; margin-top: 5px;">正在加载3D模型...</div>
        `;
        panel.appendChild(successDiv);
        
        // 自动切换到3D视图
        switchView('3d');
        return;
    }
    
    // 旧格式兼容
    if (!result.objects || result.objects.length === 0) {
        panel.innerHTML = '<div class="empty-state"><p>生成失败，请重试</p></div>';
        return;
    }
    
    // 显示3D场景
    try {
        display3DScene(result);
        console.log('3D场景已显示');
    } catch (error) {
        console.error('显示3D场景失败:', error);
    }
    
    // 添加成功提示
    const successDiv = document.createElement('div');
    successDiv.style.cssText = 'padding: 15px; background: #c6f6d5; color: #22543d; border-radius: 6px; margin-bottom: 15px; text-align: center;';
    successDiv.innerHTML = '✅ 场景生成成功！点击"3D视图"查看场景';
    panel.appendChild(successDiv);
    
    // 显示每个物体的信息
    result.objects.forEach((obj, index) => {
        const objDiv = document.createElement('div');
        objDiv.className = 'object-result';
        objDiv.innerHTML = `
            <h4>[${index}] ${obj.type_name}</h4>
            <div class="property">
                <strong>位置 (x,y,z):</strong> 
                ${obj.position.map(v => v.toFixed(3)).join(', ')}
            </div>
            <div class="property">
                <strong>尺寸 (L×W×H):</strong> 
                ${obj.size.map(v => v.toFixed(2)).join(' × ')} m
            </div>
            <div class="property">
                <strong>旋转角度:</strong> 
                ${obj.angle_deg.toFixed(1)}°
            </div>
            ${obj.jid ? `<div class="property">
                <strong>模型ID:</strong> 
                <code style="font-size: 0.8em;">${obj.jid}</code>
            </div>` : ''}
        `;
        panel.appendChild(objDiv);
    });
    
    // 切换到3D视图显示场景
    switchView('3d');
}

// 清空场景
function clearScene() {
    if (cy.nodes().length > 0) {
        if (!confirm('确定要清空所有物体和关系吗？')) {
            return;
        }
    }
    
    cy.elements().remove();
    nodeIdCounter = 0;
    updateStats();
    updateRelationSelects();
    
    // 清空3D场景
    if (typeof clear3DScene === 'function') {
        clear3DScene();
    }
    
    // 清空两个视图面板
    const viewData = document.getElementById('viewData');
    if (viewData) {
        viewData.innerHTML = `
            <div class="empty-state">
                <p>🎯 请配置场景图并点击"生成3D布局"</p>
            </div>
        `;
    }
    
    const canvas3d = document.getElementById('canvas3d');
    if (canvas3d) {
        canvas3d.innerHTML = `
            <div class="empty-state">
                <p>🎯 请配置场景图并点击"生成3D布局"</p>
            </div>
        `;
    }
}

// 切换视图（3D视图和数据视图）
function switchView(viewType) {
    const view3d = document.getElementById('view3d');
    const viewData = document.getElementById('viewData');
    const tabs = document.querySelectorAll('.tab-btn');
    
    if (!view3d || !viewData) {
        console.error('视图元素不存在');
        return;
    }
    
    // 隐藏所有视图
    view3d.style.display = 'none';
    viewData.style.display = 'none';
    
    // 移除所有active标签
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // 显示对应视图
    if (viewType === '3d') {
        view3d.style.display = 'flex';
        tabs[0]?.classList.add('active');
    } else if (viewType === 'data') {
        viewData.style.display = 'flex';
        tabs[1]?.classList.add('active');
    }
}

// 添加生成日志
function addProcessLog(type, title, content, details = null) {
    const logContainer = document.getElementById('processLogContainer');
    if (!logContainer) return;
    
    // 如果是第一条日志，清空空状态
    const emptyState = logContainer.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }
    
    const timestamp = new Date().toLocaleTimeString('zh-CN');
    const icons = {
        'step': '🚀',
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    };
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.innerHTML = `
        <div class="log-header">
            <span class="log-icon">${icons[type] || '📝'}</span>
            <span class="log-title">${title}</span>
            <span class="log-timestamp">${timestamp}</span>
        </div>
        <div class="log-content">${content}</div>
        ${details ? `<div class="log-details">${details}</div>` : ''}
    `;
    
    logContainer.appendChild(logEntry);
    
    // 自动滚动到底部
    logContainer.scrollTop = logContainer.scrollHeight;
}

// 清空生成日志
function clearProcessLog() {
    const logContainer = document.getElementById('processLogContainer');
    if (!logContainer) return;
    
    logContainer.innerHTML = `
        <div class="empty-state">
            <p>⏳ AI推理过程将在生成场景时显示</p>
        </div>
    `;
}

// 这些函数现在在 viewer3d.js 中实现
// resetCamera()
// toggleWireframe()
// exportScene()

