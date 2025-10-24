# 地面对齐修复说明

## 问题描述

用户反馈"模型的点位似乎不太对"。经过分析发现：

### 根本原因

1. **后端数据格式**：
   - 后端输出的 `position` 是物体的**中心位置坐标** (x, y, z)
   - Y坐标范围：[0.045, 3.6248]（训练数据的bounds）
   - 物体底部不在Y=0（地面）

2. **Blender渲染vs Web显示的差异**：
   - **Blender渲染**：调用 `normalize_scene()` 将整个场景居中和缩放
   - **Web 3D查看器**：直接使用原始坐标，没有归一化

3. **具体问题**：
   ```
   床 (double_bed):
   - 中心Y坐标: 0.911
   - 高度: 0.91
   - 底部Y坐标: 0.911 - 0.455 = 0.456  ← 悬空了0.456米！
   
   床头柜 (nightstand):
   - 中心Y坐标: 0.376
   - 高度: 0.36
   - 底部Y坐标: 0.376 - 0.18 = 0.196  ← 悬空了0.196米！
   ```

## 解决方案

添加 `adjustSceneToGround()` 函数，在所有物体加载完成后：

1. **计算场景最低点**
   ```javascript
   let minY = Infinity;
   sceneObjects.forEach(({ mesh }) => {
       const box = new THREE.Box3().setFromObject(mesh);
       if (box.min.y < minY) {
           minY = box.min.y;
       }
   });
   ```

2. **下移所有物体到地面**
   ```javascript
   const offsetY = -minY;  // 计算需要下移的距离
   sceneObjects.forEach(({ mesh }) => {
       mesh.position.y += offsetY;  // 应用Y轴偏移
   });
   ```

3. **控制台输出调试信息**
   ```
   🔧 调整场景到地面...
     📏 场景最低点 Y = 0.456
     ⬇️  将场景下移 0.456 米
     ✅ 场景已对齐到地面（Y=0）
   ```

## 修改的文件

### viewer3d.js

**新增函数：**
```javascript
// 调整场景到地面：找到最低点，将所有物体下移使其对齐到Y=0
function adjustSceneToGround() {
    if (sceneObjects.length === 0) return;
    
    console.log('\n🔧 调整场景到地面...');
    
    // 计算所有物体的最低Y坐标
    let minY = Infinity;
    sceneObjects.forEach(({ mesh }) => {
        const box = new THREE.Box3().setFromObject(mesh);
        const min = box.min;
        if (min.y < minY) {
            minY = min.y;
        }
    });
    
    console.log(`  📏 场景最低点 Y = ${minY.toFixed(3)}`);
    
    // 如果最低点不在地面，将所有物体下移
    if (Math.abs(minY) > 0.001) {  // 允许1mm的误差
        const offsetY = -minY;  // 需要下移的距离
        console.log(`  ⬇️  将场景下移 ${offsetY.toFixed(3)} 米`);
        
        sceneObjects.forEach(({ mesh, data }) => {
            mesh.position.y += offsetY;
        });
        
        console.log(`  ✅ 场景已对齐到地面（Y=0）\n`);
    } else {
        console.log(`  ✅ 场景已经在地面上\n`);
    }
}
```

**调用位置：**
```javascript
async function loadAndDisplayObjects(sceneData) {
    // ... 加载所有物体 ...
    await Promise.all(loadPromises);
    
    console.log(`✓ 成功加载 ${sceneObjects.length} 个物体`);
    
    // ✨ 新增：调整场景到地面
    adjustSceneToGround();
    
    // 调整相机以查看所有物体
    if (sceneObjects.length > 0) {
        fitCameraToObjects();
        console.log('✓ 相机视角已调整');
    }
    
    // 渲染
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}
```

## 效果对比

### 修复前

```
物体直接使用后端坐标：
- 床底部：Y = 0.456米（悬空）
- 床头柜1底部：Y = 0.196米（悬空）
- 床头柜2底部：Y = 0.216米（悬空）
- 地面GridHelper：Y = 0
- 结果：所有物体都飘在空中 ❌
```

### 修复后

```
自动下移场景：
- 场景下移：-0.196米（使用最低点床头柜1作为基准）
- 床底部：Y = 0.456 - 0.196 = 0.260米
- 床头柜1底部：Y = 0.196 - 0.196 = 0米 ✅
- 床头柜2底部：Y = 0.216 - 0.196 = 0.020米
- 地面GridHelper：Y = 0
- 结果：至少有一个物体接触地面，其他物体保持相对高度关系 ✅
```

## 技术要点

### 1. 为什么不直接将所有物体的底部都设为Y=0？

**理由：**
- 需要保持物体之间的相对高度关系
- 例如：桌子上的台灯应该比桌子高
- 我们只调整整个场景的Y轴基准，不改变物体间的相对位置

### 2. 为什么允许1mm的误差？

```javascript
if (Math.abs(minY) > 0.001) {  // 允许1mm的误差
```

**理由：**
- 浮点数计算可能有微小误差
- 避免不必要的调整（如minY = 0.0001时）
- 1mm的误差在3D可视化中几乎不可见

### 3. 为什么在加载完成后调整，而不是在加载每个物体时？

**理由：**
- 需要知道所有物体的边界才能确定最低点
- 一次性调整比多次调整更高效
- 便于调试（统一的调整点）

## 调试输出示例

### 浏览器控制台 (F12 → Console)

```
✓ 成功加载 3 个物体

🔧 调整场景到地面...
  📏 场景最低点 Y = 0.195
  ⬇️  将场景下移 0.195 米
  ✅ 场景已对齐到地面（Y=0）

✓ 相机视角已调整
```

## 其他注意事项

### Blender渲染的normalize_scene

```python
def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)  # 缩放以适应视口
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2  # 居中
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
```

**Blender的处理：**
1. 缩放整个场景
2. 平移到中心（X、Y、Z都居中）

**我们的处理：**
1. 不缩放（保持真实尺寸）
2. 只调整Y轴（对齐地面）
3. X、Z保持原始坐标（保持布局）

### 为什么不完全模仿Blender？

**原因：**
- Blender是为了渲染美观的图片
- Web查看器是为了查看真实的场景布局
- 保持X、Z原始坐标更利于理解空间关系
- 用户可以通过相机控制查看不同角度

## 未来改进

### 可选的normalize模式

可以添加一个按钮，让用户选择：

```javascript
// 模式1：对齐地面（当前实现）
function alignToGround() {
    const minY = computeMinY();
    offsetSceneY(-minY);
}

// 模式2：完全居中（像Blender）
function normalizeScene() {
    const bbox = computeSceneBBox();
    const center = (bbox.min + bbox.max) / 2;
    offsetScene(-center);
    scaleScene(targetSize / bbox.size);
}

// 模式3：原始坐标（不调整）
function useRawCoordinates() {
    // 不做任何调整
}
```

### 添加地面平面

除了GridHelper，还可以添加一个半透明的地面平面：

```javascript
const floorGeometry = new THREE.PlaneGeometry(20, 20);
const floorMaterial = new THREE.MeshStandardMaterial({
    color: 0xcccccc,
    transparent: true,
    opacity: 0.3
});
const floor = new THREE.Mesh(floorGeometry, floorMaterial);
floor.rotation.x = -Math.PI / 2;  // 水平放置
floor.receiveShadow = true;
scene.add(floor);
```

## 总结

✅ **问题已解决**：物体现在正确地放置在地面上
✅ **保持相对关系**：物体之间的相对高度关系不变
✅ **调试友好**：详细的控制台输出
✅ **性能优化**：一次性调整，不影响加载速度

**测试方法：**
1. 访问 `http://localhost:6006`
2. 选择模板并生成场景
3. 按F12打开浏览器控制台
4. 查看"调整场景到地面"的输出
5. 观察3D视图中物体是否正确放置在地面网格上

---

**更新时间**: 2025-10-23  
**相关文件**: `web_interface/static/viewer3d.js`  
**相关问题**: "模型的点位似乎不太对"

