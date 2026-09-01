"""生成合成工单数据集（方案 B）。

刻意埋入的"真实陷阱"，供 M2 数据审计 / M4 误差分析发现：
- 类别不均衡（billing 是大类，performance 是小类）
- `resolution` 列是人工处理完才填写的处理结果 → 数据泄漏源，分类时不可用
- 约 3% 完全重复工单、约 1% 空正文
- 约 5% 歧义工单：同一段文本（"又慢又报错"）随机标为 bug 或 performance，
  模拟人工标注不一致，制造不可约减的混淆（误差分析的主角）
- 约 2% 标题信息极少的工单（"求助"），制造短文本错误桶

用法: python src/generate_data.py   →  data/tickets.csv
固定种子，可复现；data/ 不进 git。
"""

import csv
import random
from pathlib import Path

SEED = 42
N = 2000
OUT = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"

random.seed(SEED)

CLASSES = {
    "billing": {
        "weight": 0.30,
        "subjects": ["发票问题", "退款没到账", "重复扣款", "账单金额不对", "如何开发票", "续费咨询", "多收钱了", "支付异常"],
        "bodies": [
            "我上个月申请了退款，到现在还没到账，请尽快处理",
            "这个月的账单金额和实际使用不符，多扣了钱",
            "需要开具增值税专用发票，抬头信息如下",
            "自动续费扣了款但我想取消订阅并退款",
            "支付成功但订单显示未付款，请核查",
            "同一笔订单被扣了两次款，要求退回一次",
            "发票开错了公司名称，需要重开",
            "试用结束后直接扣了年费，没有提前提醒",
        ],
        "keywords": ["发票", "退款", "账单", "扣款", "支付", "续费", "费用"],
        "resolutions": ["已退款，3个工作日内到账", "已补开发票并发送邮箱", "已核对账单并退回差额", "已取消自动续费"],
    },
    "bug": {
        "weight": 0.25,
        "subjects": ["页面报错", "功能无法使用", "导出失败", "系统崩溃", "按钮点击无反应", "数据丢失", "保存失败", "上传出错"],
        "bodies": [
            "点击导出按钮后页面直接白屏，控制台报500错误",
            "上传附件时报错，提示文件格式不支持，但文件是正常的",
            "保存的数据第二天不见了，疑似系统bug",
            "登录后偶发闪退，客户端直接崩溃",
            "提交表单卡住不动，最后提示系统异常",
            "打印预览排版全乱了，和页面显示不一致",
            "删除操作没有二次确认，误删了重要数据",
            "搜索框输入特殊字符直接报错退出",
        ],
        # "卡住""超时"与 performance 共享，是 M4 误差分析要面对的混淆源
        "keywords": ["报错", "崩溃", "白屏", "异常", "无法", "卡住", "超时"],
        "resolutions": ["已定位为缺陷，v2.3.1已修复", "已复现并提交研发团队", "已发布热修复补丁"],
    },
    "account": {
        "weight": 0.20,
        "subjects": ["无法登录", "修改绑定手机", "账号被锁定", "找回密码", "注销账号", "权限问题", "收不到验证码", "更换邮箱"],
        "bodies": [
            "忘记密码，重置邮件一直收不到",
            "更换了手机号，需要修改绑定信息",
            "多次输错密码账号被锁定了，请帮忙解锁",
            "离职了需要注销企业账号",
            "管理员权限无法分配给其他同事",
            "短信验证码收不到，没法登录",
            "账号提示在异地登录，担心被盗",
            "想把登录邮箱换成新的",
        ],
        "keywords": ["登录", "密码", "账号", "绑定", "权限", "注销", "锁定"],
        "resolutions": ["已重置密码并短信通知", "已人工核验身份后解锁", "已更新绑定信息"],
    },
    "feature_request": {
        "weight": 0.15,
        "subjects": ["希望增加功能", "建议优化", "能否支持", "功能建议", "需求反馈", "加个功能吧", "产品建议"],
        "bodies": [
            "希望报表能支持按周导出，现在只能按月",
            "建议增加深色模式，晚上使用眼睛累",
            "能否支持批量导入设备信息，一个个录入太慢了",
            "希望移动端也能查看实时数据",
            "建议搜索支持模糊匹配",
            "希望支持自定义仪表盘布局",
            "能否开放API接口让我们对接内部系统",
            "建议消息通知支持钉钉和企业微信",
        ],
        "keywords": ["希望", "建议", "能否", "支持", "增加", "优化"],
        "resolutions": ["已录入产品需求池", "已转交产品经理评估", "列入下季度规划"],
    },
    "performance": {
        "weight": 0.10,
        "subjects": ["系统很慢", "加载超时", "响应慢", "导出很慢", "页面卡顿", "越来越卡", "等待时间长"],
        "bodies": [
            "最近系统打开特别慢，首页加载要十几秒",
            "报表导出很慢，五千条数据要等五分钟以上",
            "高峰期接口响应超时，卡住不动",
            "列表页滚动卡顿明显",
            "查询响应时间比以前慢了一倍",
            "图片加载特别慢，经常转圈",
            "数据量大的时候整个系统都卡",
            "每天早上第一次登录特别慢",
        ],
        "keywords": ["慢", "超时", "卡顿", "卡住", "响应", "加载"],
        "resolutions": ["已优化慢查询并加索引", "已扩容服务器", "已定位性能瓶颈并优化"],
    },
}

# 歧义工单：同一段文本随机标 bug / performance（人工标注不一致的现实模拟）
AMBIGUOUS_BODIES = [
    "导出报表又慢最后还报错了",
    "系统卡住不动，半天没反应还提示异常",
    "操作超时，不知道是崩了还是单纯慢",
]
AMBIGUOUS_CLASSES = ["bug", "performance"]

# 跨类共享从句：同一措辞合理地属于两个类别，标注 50/50 —— 任何模型都拿不满分，
# 这是"不可约减错误"，M4 误差分析要把它识别为标注规范问题而非模型问题
SHARED_ISSUES = [
    ("导出一直没反应，等了很久还失败", ("bug", "performance")),
    ("提交后系统一直没响应", ("bug", "performance")),
    ("登录特别慢，有时候还登不上", ("performance", "account")),
    ("页面一直转圈，最后提示失败", ("performance", "bug")),
    ("扣款后功能还是用不了", ("billing", "bug")),
    ("收不到邮件，重置不了", ("account", "bug")),
    ("希望能快点修复，太影响使用了", ("feature_request", "bug")),
    ("导入数据又慢还经常中断", ("performance", "bug")),
]

# 30%→45% 的工单用与类别无关的通用标题（"反馈问题"），剥掉标题这个捷径
GENERIC_SUBJECTS = ["反馈一个问题", "使用咨询", "帮忙看看", "系统问题", "客户反馈"]

# 正文组合件：opening + issue + detail(按类) + closing，组合数 >> 样本数，
# 保证 val 里必然出现训练集没见过的措辞 —— 逼模型真正泛化，而不是背模板
OPENINGS = ["你好，", "麻烦看一下：", "我这边遇到个问题：", "反馈一下：", "客服你好，", ""]
CLOSINGS = ["请尽快处理。", "谢谢。", "望回复。", "很急，麻烦了。", ""]

DETAILS = {
    "billing": ["金额是328元。", "已经两周了。", "订单号20260812。", "涉及三张发票。", ""],
    "bug": ["用的是最新版客户端。", "每次必现。", "浏览器和客户端都有这个问题。", "大概从昨天开始。", ""],
    "account": ["手机号是138开头的。", "公司账号下有二十多个成员。", "试了好几次都不行。", ""],
    "feature_request": ["我们团队五十多人都有这个需求。", "竞品都有这个功能。", "可以付费升级。", ""],
    "performance": ["我们这边网络没问题。", "数据量大概五万条。", "高峰期尤其明显。", ""],
}

# 25%→45% 的正文被"串味"：拼上含其他类关键词的从句，模拟真实工单的混杂表达
CONTAMINATION = [
    "，而且一直报错无法提交",
    "，响应也特别慢",
    "，能不能顺便增加导出功能",
    "，另外密码也改不了",
    "，账单金额好像也不对",
]

CHANNELS = ["web", "email", "phone", "api"]
PRIORITY_BY_CLASS = {
    "bug": ["high", "high", "medium", "low"],
    "performance": ["high", "medium", "medium", "low"],
    "billing": ["medium", "medium", "low", "high"],
    "account": ["medium", "low", "high", "low"],
    "feature_request": ["low", "low", "medium", "high"],
}

TINY_SUBJECTS = ["求助", "问题", "看一下", "急"]


def make_ticket(i: int) -> dict:
    r = random.random()
    if r < 0.05:
        # 歧义工单：措辞歧义 + 通用标题 + 无细节，信号被彻底抽掉，标注 50/50
        cls = random.choice(AMBIGUOUS_CLASSES)
        return _row(i, cls, random.choice(GENERIC_SUBJECTS), random.choice(AMBIGUOUS_BODIES))
    if r < 0.13:
        # 共享从句工单：同样抽掉标题和细节信号，制造不可约减混淆
        issue, pair = random.choice(SHARED_ISSUES)
        cls = random.choice(pair)
        return _row(i, cls, random.choice(GENERIC_SUBJECTS), random.choice(OPENINGS) + issue)

    cls = random.choices(list(CLASSES), weights=[c["weight"] for c in CLASSES.values()])[0]
    subject = random.choice(CLASSES[cls]["subjects"])
    body = random.choice(OPENINGS) + random.choice(CLASSES[cls]["bodies"]) + random.choice(DETAILS[cls]) + random.choice(CLOSINGS)
    if random.random() < 0.5:
        body += "，" + random.choice(CLASSES[cls]["keywords"]) + "相关问题"
    if random.random() < 0.45:
        body += random.choice(CONTAMINATION)  # 串味从句：引入其他类关键词
    if random.random() < 0.02:
        subject = random.choice(TINY_SUBJECTS)  # 短文本工单
    elif random.random() < 0.45:
        subject = random.choice(GENERIC_SUBJECTS)  # 通用标题，标题失去类别信号
    return _row(i, cls, subject, body)


def _row(i: int, cls: str, subject: str, body: str) -> dict:
    if random.random() < 0.01:
        body = ""  # 空正文
    return {
        "ticket_id": f"T{i:05d}",
        "subject": subject,
        "description": body,
        "channel": random.choice(CHANNELS),
        "priority": random.choice(PRIORITY_BY_CLASS[cls]),
        "type": cls,
        # 泄漏列：处理结果，由标签决定 —— 分类任务输入时不可用
        "resolution": random.choice(CLASSES[cls]["resolutions"]),
    }
    if random.random() < 0.01:
        body = ""  # 空正文
    return {
        "ticket_id": f"T{i:05d}",
        "subject": subject,
        "description": body,
        "channel": random.choice(CHANNELS),
        "priority": random.choice(PRIORITY_BY_CLASS[cls]),
        "type": cls,
        # 泄漏列：处理结果，由标签决定 —— 分类任务输入时不可用
        "resolution": random.choice(CLASSES[cls]["resolutions"]),
    }


def main() -> None:
    rows = [make_ticket(i) for i in range(1, N + 1)]
    for _ in range(int(N * 0.03)):  # 3% 完全重复（改 ticket_id）
        dup = dict(random.choice(rows))
        dup["ticket_id"] = f"T{len(rows) + 1:05d}"
        rows.append(dup)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"生成 {len(rows)} 条 → {OUT}")


if __name__ == "__main__":
    main()
