# Codemagic iOS 构建环境 — 搭建指南

> **免费 Apple ID 即可使用，不需要 $99/年开发者账号。**

---

## 架构概览

```
Windows 开发                      Codemagic 云 Mac               iPhone 真机
───────────                      ──────────────                ──────────

flutter run -d windows ──→ 日常开发 + 调试
  • Hot Reload ✅
  • DevTools ✅
  • 断点 ✅

git push ────────────────→ flutter build ipa --debug
                           用免费 Apple ID 签名
                           .ipa 7天有效            ──→ 安装到 iPhone
                                                       手动测试验证
```

## 第一步：注册 Codemagic（免费）

1. 打开 https://codemagic.io
2. 用 GitHub 登录（或用 GitLab/Bitbucket）
3. 免费额度：**500 分钟/月**（单次 iOS 构建约 10-15 分钟）

## 第二步：准备 Apple ID（免费）

**用你现有的 Apple ID 就行**，不需要注册开发者：

1. 打开 https://appleid.apple.com 确认你的 Apple ID 可用
2. 如果没有 → 免费注册一个
3. iPhone 上登录同一个 Apple ID

> 免费 Apple ID 的限制：
> - 签名 7 天后过期（需重新构建安装）
> - 最多 3 个设备
> - 不能上架 App Store（需要时再升级 $99/年）

## 第三步：连接项目到 Codemagic

1. 先把项目推送到远程仓库：
   ```bash
   # 如果还没推送到 GitHub/Gitee：
   git remote add origin https://github.com/你的用户名/Who_are_U.git
   git push -u origin main
   ```

2. Codemagic 后台 → **Add Application** → 选择你的仓库
3. 项目类型选 **Flutter App (via YAML)**
4. Codemagic 会自动检测 `codemagic.yaml`

## 第四步：配置免费签名

1. Codemagic 后台 → 你的 App → **App Settings → Code signing**
2. **iOS code signing** 选择 **Automatic**
3. 登录你的 Apple ID（普通 Apple ID，不是开发者账号）
4. Codemagic 自动创建免费 Provisioning Profile

## 第五步：设置通知邮箱

Codemagic 后台 → **App Settings → Environment Variables**：

| 变量 | 值 |
|------|-----|
| `NOTIFY_EMAIL` | `你的邮箱@example.com` |

## 第六步：触发首次构建

```bash
# 随便哪个分支 push 都会触发
git add .
git commit -m "feat: 测试 Codemagic iOS 构建"
git push
```

构建完成后（约 10-15 分钟）：
1. Codemagic 会发邮件通知
2. 从构建页面下载 `.ipa` 文件
3. 安装到 iPhone（见下方）

## 第七步：安装 .ipa 到 iPhone

### 方法 1：Codemagic 直接安装（推荐）

构建成功后，Codemagic 页面有 **Install** 按钮 → 扫描二维码 → 安装。

### 方法 2：Apple Configurator（Windows）

1. 在 Microsoft Store 安装 [Apple Devices](https://apps.microsoft.com/store/detail/apple-devices/9NP83LWLPZ9K)
2. USB 连接 iPhone → 拖入 .ipa → 安装

### 方法 3：爱思助手

1. 下载爱思助手 → USB 连接 iPhone
2. 应用游戏 → 导入安装 → 选择 .ipa

---

## 日常工作流

```
┌─ 写代码 ──→ flutter run -d windows  调试确认 ──→ git push ──→ Codemagic 构建
│                                                                    │
└────────────────────── 回到第1步 继续开发 ←────── iPhone 测试 ←─────┘
```

### 何时需要 iOS 真机测试

Flutter 跨平台一致性极高，以下场景才需要 iPhone 验证：

- 安全区域 / 刘海屏 / 灵动岛适配
- iOS 特有手势（侧滑返回、长按菜单）
- 键盘弹出/收起布局
- 原生支付（in_app_purchase StoreKit）
- Metal 渲染性能

其他 95% 的情况，Windows 上调试的结果和 iOS 一致。

---

## 后续升级（需要上架 App Store 时）

```bash
# 只需 3 步：
# 1. 注册 Apple Developer ($99/年)
# 2. Codemagic → Integrations → 连接 App Store Connect
# 3. Push release 分支 → 自动上传 TestFlight
```
