// Three.js 3D场景查看器

let scene, camera, renderer, controls;
let sceneObjects = [];
let labelRenderer;

// 初始化3D场景
function init3DScene() {
    const container = document.getElementById('canvas3d');
    if (!container) return;
    
    // 清空容器
    container.innerHTML = '';
    
    // 创建场景
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a202c);
    
    // 创建相机
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);
    
    // 创建渲染器
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setClearColor(0xffffff, 1);  // 设置背景为白色
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    
    // 添加轨道控制器
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 50;
    
    // 添加网格地面
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    scene.add(gridHelper);
    
    // 添加坐标轴
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);
    
    // 添加光照
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.camera.left = -15;
    directionalLight.shadow.camera.right = 15;
    directionalLight.shadow.camera.top = 15;
    directionalLight.shadow.camera.bottom = -15;
    scene.add(directionalLight);
    
    // 添加点光源
    const pointLight = new THREE.PointLight(0xffffff, 0.5);
    pointLight.position.set(0, 5, 0);
    scene.add(pointLight);
    
    // 启动动画循环
    animate();
    
    // 响应窗口大小变化
    window.addEventListener('resize', onWindowResize);
}

// 动画循环
function animate() {
    requestAnimationFrame(animate);
    
    if (controls) {
        controls.update();
    }
    
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// 窗口大小变化
function onWindowResize() {
    const container = document.getElementById('canvas3d');
    if (!container || !camera || !renderer) return;
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// 清空场景中的物体
function clear3DScene() {
    if (!scene) return;
    
    console.log(`清空场景，当前有 ${sceneObjects.length} 个物体`);
    
    sceneObjects.forEach(obj => {
        // 移除网格
        if (obj.mesh) {
            scene.remove(obj.mesh);
            // 释放几何体和材质
            if (obj.mesh.geometry) obj.mesh.geometry.dispose();
            if (obj.mesh.material) obj.mesh.material.dispose();
        }
        // 移除标签
        if (obj.label) {
            scene.remove(obj.label);
            if (obj.label.material && obj.label.material.map) {
                obj.label.material.map.dispose();
            }
            if (obj.label.material) obj.label.material.dispose();
        }
    });
    
    sceneObjects = [];
    console.log('场景已清空');
}

// 物体颜色映射（根据类型）
const objectColors = {
    'bed': 0x8b4513,          // 棕色
    'nightstand': 0xdeb887,   // 浅棕色
    'wardrobe': 0xa0522d,     // 深棕色
    'desk': 0xcd853f,         // 秘鲁色
    'chair': 0xd2691e,        // 巧克力色
    'bookshelf': 0x8b7355,    // 浅棕
    'sofa': 0x4169e1,         // 皇家蓝
    'table': 0xdaa520,        // 金黄色
    'cabinet': 0x556b2f,      // 橄榄绿
    'armchair': 0x6b8e23,     // 黄绿色
    'default': 0x888888       // 灰色
};

// 获取物体颜色
function getObjectColor(typeName) {
    for (const key in objectColors) {
        if (typeName.toLowerCase().includes(key)) {
            return objectColors[key];
        }
    }
    return objectColors.default;
}

// 添加3D物体到场景
async function add3DObject(objectData) {
    const { type_name, position, size, angle, jid } = objectData;
    
    console.log(`\n========== 添加物体 ==========`);
    console.log(`类型: ${type_name}`);
    console.log(`位置 (x,y,z): [${position[0].toFixed(3)}, ${position[1].toFixed(3)}, ${position[2].toFixed(3)}]`);
    console.log(`尺寸 (L×W×H): [${size[0].toFixed(2)}, ${size[1].toFixed(2)}, ${size[2].toFixed(2)}]`);
    console.log(`旋转角度: ${(angle * 180 / Math.PI).toFixed(1)}° (${angle.toFixed(4)} rad)`);
    console.log(`模型ID (JID): ${jid || '无'}`);
    
    let mesh;
    
    // 如果有jid，尝试加载真实模型
    if (jid) {
        try {
            mesh = await loadRealModel(jid, size, position, angle);
            console.log(`✓ 成功加载模型: ${jid}`);
        } catch (error) {
            console.warn(`加载模型失败，使用简单立方体: ${error.message}`);
            mesh = createSimpleBox(type_name, size, position, angle);
        }
    } else {
        // 没有jid，使用简单立方体
        mesh = createSimpleBox(type_name, size, position, angle);
    }
    
    // 添加阴影
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.traverse(child => {
        if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });
    
    // 添加到场景
    scene.add(mesh);
    
    // 创建文字标签
    const labelText = jid ? `${type_name}\n(${jid.substring(0, 8)})` : type_name;
    const label = createTextLabel(labelText, position, size);
    scene.add(label);
    
    // 保存引用
    sceneObjects.push({ mesh, label, data: objectData });
}

// 创建简单立方体（作为后备）
function createSimpleBox(type_name, size, position, angle) {
    // size = [x, y, z]，其中 y 是高度
    // BoxGeometry(width, height, depth) = (X, Y, Z)
    const geometry = new THREE.BoxGeometry(size[0], size[1], size[2]);
    const color = getObjectColor(type_name);
    const material = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.7,
        metalness: 0.2,
        transparent: true,
        opacity: 0.8
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    // 位置：Y坐标使用 size[1] / 2（高度的一半）
    mesh.position.set(position[0], position[1], position[2]);
    mesh.rotation.y = angle;
    
    return mesh;
}

// 加载真实的OBJ模型
function loadRealModel(jid, size, position, angle) {
    return new Promise((resolve, reject) => {
        const mtlLoader = new THREE.MTLLoader();
        const objLoader = new THREE.OBJLoader();
        
        // 设置基础URL
        const baseUrl = `/api/model/${jid}/`;
        
        // 首先加载MTL材质文件
        mtlLoader.load(
            baseUrl + 'model.mtl',
            (materials) => {
                materials.preload();
                objLoader.setMaterials(materials);
                
                // 然后加载OBJ模型
                objLoader.load(
                    baseUrl + 'raw_model.obj',
                    (object) => {
                        // 参考 src/utils/visualize.py 的变换逻辑：
                        // 1. 缩放：tr_mesh.vertices *= scale
                        // 2. 居中：tr_mesh.vertices -= (tr_mesh.bounds[0] + tr_mesh.bounds[1]) / 2.
                        // 3. 旋转+平移：tr_mesh.vertices = tr_mesh.vertices.dot(R) + translation
                        
                        console.log(`  🔧 开始应用变换...`);
                        
                        // 步骤1: 计算原始模型的边界框和缩放比例
                        let box = new THREE.Box3().setFromObject(object);
                        const modelSize = box.getSize(new THREE.Vector3());
                        console.log(`  📏 原始模型尺寸: [${modelSize.x.toFixed(3)}, ${modelSize.y.toFixed(3)}, ${modelSize.z.toFixed(3)}]`);
                        
                        // 计算缩放比例 - 使用预测的尺寸
                        const scaleX = size[0] / modelSize.x;
                        const scaleY = size[1] / modelSize.y;  // Y是高度
                        const scaleZ = size[2] / modelSize.z;
                        const uniformScale = Math.min(scaleX, scaleY, scaleZ);
                        console.log(`  🔍 缩放比例: X=${scaleX.toFixed(3)}, Y=${scaleY.toFixed(3)}, Z=${scaleZ.toFixed(3)}`);
                        console.log(`  ✓ 使用统一缩放: ${uniformScale.toFixed(3)}`);
                        
                        // 步骤2: 应用缩放
                        object.scale.set(uniformScale, uniformScale, uniformScale);
                        
                        // 步骤3: 重新计算缩放后的边界框，并居中
                        box.setFromObject(object);
                        const scaledCenter = box.getCenter(new THREE.Vector3());
                        const scaledSize = box.getSize(new THREE.Vector3());
                        console.log(`  📦 缩放后中心: [${scaledCenter.x.toFixed(3)}, ${scaledCenter.y.toFixed(3)}, ${scaledCenter.z.toFixed(3)}]`);
                        console.log(`  📦 缩放后尺寸: [${scaledSize.x.toFixed(3)}, ${scaledSize.y.toFixed(3)}, ${scaledSize.z.toFixed(3)}]`);
                        
                        // 创建Group来应用后续变换
                        const group = new THREE.Group();
                        
                        // 将缩放后的模型移到原点（居中）
                        object.position.set(-scaledCenter.x, -scaledCenter.y, -scaledCenter.z);
                        group.add(object);
                        console.log(`  🎯 模型已居中到原点`);
                        
                        // 步骤4: 应用旋转（绕Y轴）
                        group.rotation.y = angle;
                        console.log(`  🔄 应用旋转: ${(angle * 180 / Math.PI).toFixed(1)}°`);
                        
                        // 步骤5: 应用平移到目标位置
                        group.position.set(position[0], position[1], position[2]);
                        console.log(`  📍 移动到目标位置: [${position[0].toFixed(3)}, ${position[1].toFixed(3)}, ${position[2].toFixed(3)}]`);
                        console.log(`  ✅ 变换完成！\n`);
                        
                        resolve(group);
                    },
                    (xhr) => {
                        console.log(`模型加载进度: ${(xhr.loaded / xhr.total * 100).toFixed(2)}%`);
                    },
                    (error) => {
                        reject(new Error(`OBJ加载失败: ${error}`));
                    }
                );
            },
            (xhr) => {
                // MTL加载进度
            },
            (error) => {
                // MTL加载失败，尝试仅加载OBJ
                console.warn('MTL加载失败，尝试仅加载OBJ');
                objLoader.load(
                    baseUrl + 'raw_model.obj',
                    (object) => {
                        // 应用默认材质
                        object.traverse(child => {
                            if (child.isMesh) {
                                child.material = new THREE.MeshStandardMaterial({
                                    color: 0x888888,
                                    roughness: 0.7,
                                    metalness: 0.3
                                });
                            }
                        });
                        
                        // 使用相同的变换逻辑
                        let box = new THREE.Box3().setFromObject(object);
                        const modelSize = box.getSize(new THREE.Vector3());
                        
                        const scaleX = size[0] / modelSize.x;
                        const scaleY = size[1] / modelSize.y;
                        const scaleZ = size[2] / modelSize.z;
                        const uniformScale = Math.min(scaleX, scaleY, scaleZ);
                        
                        object.scale.set(uniformScale, uniformScale, uniformScale);
                        
                        box.setFromObject(object);
                        const scaledCenter = box.getCenter(new THREE.Vector3());
                        
                        const group = new THREE.Group();
                        object.position.set(-scaledCenter.x, -scaledCenter.y, -scaledCenter.z);
                        group.add(object);
                        
                        group.rotation.y = angle;
                        group.position.set(position[0], position[1], position[2]);
                        
                        resolve(group);
                    },
                    undefined,
                    (error) => {
                        reject(new Error(`OBJ加载失败: ${error}`));
                    }
                );
            }
        );
    });
}

// 创建文字标签
function createTextLabel(text, position, size) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 80;  // 增加高度以支持多行
    
    // 绘制背景
    context.fillStyle = 'rgba(0, 0, 0, 0.7)';
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制文字（支持多行）
    context.font = 'Bold 18px Arial';
    context.fillStyle = 'white';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    
    const lines = text.split('\n');
    const lineHeight = 24;
    const startY = canvas.height / 2 - ((lines.length - 1) * lineHeight) / 2;
    
    lines.forEach((line, i) => {
        context.fillText(line, canvas.width / 2, startY + i * lineHeight);
    });
    
    // 创建纹理
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    
    // 设置位置（在物体上方）
    sprite.position.set(position[0], size[2] + 0.6, position[2]);
    sprite.scale.set(2.5, 0.8, 1);  // 调整大小以适应更多文字
    
    return sprite;
}

// 显示3D场景
async function display3DScene(sceneData) {
    console.log('开始显示3D场景:', sceneData);
    
    if (!scene) {
        console.log('初始化3D场景...');
        init3DScene();
        // 等待场景初始化完成
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // 显示加载提示
    showLoadingIndicator(sceneData.objects ? sceneData.objects.length : 1);
    
    try {
        // 如果有scene_url，说明是Blender导出的场景
        if (sceneData.scene_url) {
            await loadBlenderScene(sceneData.scene_url);
        } else {
            // 否则使用之前的逐个物体加载方式
            await loadAndDisplayObjects(sceneData);
        }
    } catch (error) {
        console.error('❌ 加载场景失败:', error);
        alert('加载场景失败: ' + error.message);
    } finally {
        // 隐藏加载提示
        hideLoadingIndicator();
    }
}

// 显示加载提示
function showLoadingIndicator(count) {
    const canvas3d = document.getElementById('canvas3d');
    let indicator = document.getElementById('modelLoadingIndicator');
    
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'modelLoadingIndicator';
        indicator.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            font-size: 16px;
            z-index: 1000;
            text-align: center;
        `;
        canvas3d.appendChild(indicator);
    }
    
    indicator.innerHTML = `
        <div>🔄 正在加载 ${count} 个3D模型...</div>
        <div style="font-size: 12px; margin-top: 10px; opacity: 0.8;">这可能需要几秒钟</div>
    `;
    indicator.style.display = 'block';
}

// 隐藏加载提示
function hideLoadingIndicator() {
    const indicator = document.getElementById('modelLoadingIndicator');
    if (indicator) {
        indicator.style.display = 'none';
    }
}

// 加载并显示所有物体
async function loadAndDisplayObjects(sceneData) {
    // 清空旧物体
    clear3DScene();
    
    // 检查数据有效性
    if (!sceneData.objects || sceneData.objects.length === 0) {
        console.warn('没有物体数据');
        return;
    }
    
    console.log(`开始加载 ${sceneData.objects.length} 个物体...`);
    
    // 并行加载所有物体
    const loadPromises = sceneData.objects.map(async (obj, index) => {
        try {
            await add3DObject(obj);
            console.log(`✓ 已加载 ${index + 1}/${sceneData.objects.length}: ${obj.type_name}`);
        } catch (error) {
            console.error(`✗ 加载物体失败 ${obj.type_name}:`, error);
        }
    });
    
    // 等待所有物体加载完成
    await Promise.all(loadPromises);
    
    console.log(`✓ 成功加载 ${sceneObjects.length} 个物体`);
    
    // 所有物体添加完成后，调整相机视角
    if (sceneObjects.length > 0) {
        fitCameraToObjects();
        console.log('✓ 相机视角已调整');
    }
    
    // 强制渲染一次
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// 调整相机以适应所有物体
function fitCameraToObjects() {
    if (sceneObjects.length === 0) return;
    
    // 计算边界框
    const box = new THREE.Box3();
    sceneObjects.forEach(obj => {
        box.expandByObject(obj.mesh);
    });
    
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    
    // 计算合适的相机距离
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 2.5; // 放大一些以确保全部可见
    
    // 设置相机位置
    camera.position.set(
        center.x + cameraZ * 0.5,
        center.y + cameraZ * 0.8,
        center.z + cameraZ * 0.5
    );
    
    // 设置控制器目标
    controls.target.copy(center);
    controls.update();
}

// 重置相机视角
function resetCamera() {
    if (!camera || !controls) return;
    
    camera.position.set(8, 8, 8);
    controls.target.set(0, 0, 0);
    controls.update();
}

// 切换线框模式
let wireframeMode = false;
function toggleWireframe() {
    wireframeMode = !wireframeMode;
    
    sceneObjects.forEach(obj => {
        obj.mesh.material.wireframe = wireframeMode;
    });
}

// 导出场景截图
function exportScene() {
    if (!renderer) {
        alert('3D场景未初始化');
        return;
    }
    
    // 渲染当前场景
    renderer.render(scene, camera);
    
    // 获取截图
    const imgData = renderer.domElement.toDataURL('image/png');
    
    // 下载
    const link = document.createElement('a');
    link.download = 'scene_3d_view.png';
    link.href = imgData;
    link.click();
}

// 加载合并的场景文件（单个OBJ包含所有物体）
async function loadMergedScene(baseUrl, filename) {
    console.log('📦 加载合并场景文件:', filename);
    
    try {
        const objLoader = new THREE.OBJLoader();
        const mtlLoader = new THREE.MTLLoader();
        
        // 检查MTL文件
        const mtlName = filename.replace('.obj', '.mtl');
        
        let materials = null;
        try {
            materials = await new Promise((resolve, reject) => {
                mtlLoader.setResourcePath(baseUrl + '/');
                mtlLoader.setPath(baseUrl + '/');
                mtlLoader.load(
                    mtlName,
                    resolve,
                    undefined,
                    (error) => {
                        console.warn(`⚠️  加载MTL失败:`, error);
                        resolve(null);
                    }
                );
            });
            
            if (materials) {
                materials.preload();
                objLoader.setMaterials(materials);
                console.log('✓ MTL材质已加载');
            }
        } catch (error) {
            console.warn('⚠️  MTL加载出错，使用默认材质:', error);
        }
        
        // 加载OBJ
        const object = await new Promise((resolve, reject) => {
            objLoader.load(
                `${baseUrl}/${filename}`,
                resolve,
                (progress) => {
                    console.log(`加载进度: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
                },
                reject
            );
        });
        
        // 设置阴影
        object.castShadow = true;
        object.receiveShadow = true;
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                child.castShadow = true;
                child.receiveShadow = true;
                
                // 确保材质正确设置
                if (child.material) {
                    child.material.side = THREE.FrontSide;
                }
            }
        });
        
        scene.add(object);
        sceneObjects.push({mesh: object});
        
        console.log('✓ 合并场景加载完成');
        
        // 调整相机
        fitCameraToObjects();
        renderer.render(scene, camera);
        
    } catch (error) {
        console.error('❌ 加载合并场景失败:', error);
        throw error;
    }
}

// 加载Blender导出的场景
async function loadBlenderScene(sceneUrl) {
    console.log('📦 加载Blender导出的场景:', sceneUrl);
    
    try {
        // 清除之前的物体
        clear3DScene();
        
        // 获取场景的OBJ文件列表
        const response = await fetch(sceneUrl);
        if (!response.ok) {
            throw new Error(`获取场景文件列表失败: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('📄 场景文件列表:', data);
        
        if (!data.files || data.files.length === 0) {
            throw new Error('场景中没有OBJ文件');
        }
        
        // 设置基础URL
        const baseUrl = data.base_url;
        
        // 优先检查是否有合并的场景文件
        const mergedFile = data.files.find(f => f.name === 'scene_merged.obj');
        if (mergedFile) {
            console.log('🎯 发现合并场景文件，使用单一加载方式');
            await loadMergedScene(baseUrl, mergedFile.name);
            return;
        }
        
        // 收集所有OBJ文件（排除bbox文件）
        const objFiles = data.files.filter(f => 
            f.name.endsWith('.obj') && 
            f.name.startsWith('object_')  // 只加载object_XX.obj，不加载bbox
        );
        
        console.log(`📦 找到 ${objFiles.length} 个物体文件`);
        
        // 并行加载所有模型
        const loadPromises = objFiles.map(async (file) => {
            console.log(`📦 加载物体: ${file.name}`);
            
            try {
                const objLoader = new THREE.OBJLoader();
                const mtlLoader = new THREE.MTLLoader();
                
                // 检查是否有对应的MTL文件
                const mtlName = file.name.replace('object_', 'material_').replace('.obj', '.mtl');
                const mtlFile = data.files.find(f => f.name === mtlName);
                
                console.log(`  → 对应MTL文件: ${mtlName}`, mtlFile ? '✓ 找到' : '✗ 未找到');
                
                let object;
                
                if (mtlFile) {
                    // 先加载MTL
                    const materials = await new Promise((resolve, reject) => {
                        // 设置资源路径，用于加载纹理
                        mtlLoader.setResourcePath(baseUrl + '/');
                        mtlLoader.setPath(baseUrl + '/');
                        mtlLoader.load(
                            mtlName,
                            (mtl) => {
                                console.log(`  ✓ MTL加载成功: ${mtlName}`);
                                resolve(mtl);
                            },
                            undefined,
                            (error) => {
                                console.warn(`  ⚠️  加载MTL失败:`, error);
                                resolve(null);
                            }
                        );
                    });
                    
                    if (materials) {
                        materials.preload();
                        objLoader.setMaterials(materials);
                        console.log(`  ✓ 材质已预加载`);
                    }
                }
                
                // 加载OBJ
                object = await new Promise((resolve, reject) => {
                    objLoader.load(
                        `${baseUrl}/${file.name}`,
                        resolve,
                        undefined,
                        reject
                    );
                });
                
                // Blender导出的场景已经处理好了位置和缩放，直接添加
                object.castShadow = true;
                object.receiveShadow = true;
                
                // 为所有子mesh设置阴影
                object.traverse((child) => {
                    if (child instanceof THREE.Mesh) {
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                
                scene.add(object);
                sceneObjects.push({mesh: object});  // 保持格式一致
                console.log(`✓ 成功加载: ${file.name}`);
                
                return object;
                
            } catch (error) {
                console.warn(`⚠️  加载 ${file.name} 失败:`, error);
                return null;
            }
        });
        
        // 等待所有模型加载完成
        await Promise.all(loadPromises);
        
        console.log('✓ Blender场景加载完成，共', sceneObjects.length, '个物体');
        
        // 调整相机以查看整个场景
        if (sceneObjects.length > 0) {
            fitCameraToObjects();
        }
        
        // 强制渲染
        renderer.render(scene, camera);
        
    } catch (error) {
        console.error('❌ 加载Blender场景失败:', error);
        throw error;
    }
}

// 清理资源
function dispose3DScene() {
    if (renderer) {
        renderer.dispose();
    }
    window.removeEventListener('resize', onWindowResize);
}

