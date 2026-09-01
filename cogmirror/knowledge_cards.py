"""知识点小卡片（按 topic 的概念教学文案，答错后可展开学习）.

静态文案（确定性、零依赖），结构供 web UI 渲染：
    {"title": str, "blocks": [("p"|"code", text), ...]}

写法约束：先给心智模型再给代码示例，最后点常见坑——卡片是
「答错之后的学习材料」，不是文档，每张控制在可 1 分钟读完。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Block = Tuple[str, str]

CARDS: Dict[str, Dict[str, object]] = {
    "python.variables": {
        "title": "变量与赋值",
        "blocks": [
            ("p", "Python 的变量不是「装值的盒子」，而是「贴在对象上的名字标签」。赋值 x = 3 做两件事：先算右边，再把名字 x 绑到结果上。所以 x = x + 1 完全合法——先算出 4，再让 x 指向 4。"),
            ("code", "x = 3\nx = x + 1   # 先算右边 3+1=4，再把 x 绑到 4\n# 注意：= 是赋值，== 才是比较"),
            ("p", "可变对象（列表/字典）赋值不复制。b = a 之后两个名字指向同一个列表，改 b 就是改 a；想独立就要显式复制（a[:] 或 list(a)）。"),
            ("code", "a = [1, 2]\nb = a          # b 和 a 是同一个列表\nb.append(3)    # a 也变成 [1, 2, 3]\n\nc = a[:]       # c 是新列表（浅拷贝）\nc.append(4)    # a 不受影响"),
            ("p", "易混点：b = a + [3] 创建新列表再绑定 b（a 不变）；b += [3] 是原地修改（a 跟着变）。交换两个变量用多重赋值：a, b = b, a。"),
        ],
    },
    "python.loops": {
        "title": "循环",
        "blocks": [
            ("p", "for 适合「已知次数 / 遍历序列」，while 适合「满足条件就继续」。range(start, stop, step) 是左闭右开：含 start、不含 stop。"),
            ("code", "for i in range(5):      # 0 1 2 3 4（不含 5）\n    print(i)\nfor i in range(1, 5, 2): # 1 3（步长 2，不含 5）\n    print(i)"),
            ("p", "while 死循环的头号原因：循环体里忘了更新条件变量。循环三要素——初始化、终止条件、状态更新——缺一不可。"),
            ("code", "i = 0\nwhile i < 5:\n    print(i)\n    i += 1   # 漏了这行 = 死循环"),
            ("p", "两个最常用的模式：累加器（total 从 0 开始逐个加）和打擂台找最大值（用第一个元素当初始擂主，逐个比较更新）。"),
            ("code", "total = 0\nfor i in range(1, n + 1):\n    total += i   # 累加器模式\n\nbiggest = nums[0]\nfor n in nums:\n    if n > biggest:\n        biggest = n   # 打擂台模式"),
        ],
    },
    "python.functions": {
        "title": "函数",
        "blocks": [
            ("p", "def 定义函数，return 把结果交还给调用方。没有 return 的函数返回 None。print 只是往屏幕输出，不产生返回值——y = f(3) 想拿到 6，函数体必须用 return，光 print 拿到的是 None。"),
            ("code", "def f(x):\n    print(x * 2)   # 只显示，不返回\n\ndef g(x):\n    return x * 2   # 返回值\n\ny = f(3)   # y 是 None\ny = g(3)   # y 是 6"),
            ("p", "参数传的是对象引用：把列表传进函数，函数里 append 会改到调用方的原列表。想不改原列表，先复制一份再操作。"),
            ("p", "经典坑——可变默认参数：def f(x, lst=[]) 里的 [] 只在定义时创建一次，所有调用共享同一个列表，元素会跨调用累积。正确写法用 None 当哨兵。"),
            ("code", "def f(x, lst=None):\n    if lst is None:\n        lst = []    # 每次调用都是新列表\n    lst.append(x)\n    return lst"),
            ("p", "函数是「一等公民」：可以赋给变量、当参数传给别的函数（如 apply_twice(f, x) = f(f(x))）。"),
        ],
    },
    "python.recursion": {
        "title": "递归",
        "blocks": [
            ("p", "递归的两个要素缺一不可：① 函数调用自身（把问题化成更小的同类子问题）；② 基准情形（base case，小到可以直接给答案、不再递归）。缺基准情形 = 无限递归，直到超过默认深度（约 1000 层）抛 RecursionError。"),
            ("code", "def factorial(n):\n    if n <= 1:\n        return 1                 # 基准情形\n    return n * factorial(n - 1)  # 化归：n! = n * (n-1)!"),
            ("p", "理解递归的两步：先「压栈」——每次调用压一层，参数越来越小直到基准情形；再「弹栈」——从最底层的答案逐层往回算。递归和循环可以互相改写，但树形结构（每个子树还是树）用递归最自然。"),
            ("code", "def reverse_str(s):\n    if len(s) <= 1:\n        return s                        # 基准：单个字符\n    return reverse_str(s[1:]) + s[0]    # 反转剩余部分，再把首字符挪到最后"),
        ],
    },
    "python.scope": {
        "title": "作用域",
        "blocks": [
            ("p", "Python 查名字按 LEGB 顺序：Local（函数内）→ Enclosing（外层函数）→ Global（模块级）→ Builtin（内置）。函数内的赋值会创建局部变量——哪怕赋值写在引用之后，整个函数里这个名字都被当作局部（print(x) 后跟 x = 5 会报 UnboundLocalError）。"),
            ("p", "想在函数内修改全局变量要显式声明 global；想改外层函数的变量要 nonlocal。两者都是「我要改外面的，不是新建局部的」。"),
            ("code", "count = 0\ndef step():\n    global count   # 不声明的话 count += 1 会创建局部变量并报错\n    count += 1\n    return count"),
            ("p", "闭包：内层函数「记住」外层函数的变量，外层返回内层函数后这份记忆还在。只读不需要声明；要修改用 nonlocal。"),
            ("code", "def make_counter():\n    count = 0\n    def counter():\n        nonlocal count\n        count += 1\n        return count\n    return counter\n\nc = make_counter()\nc()  # 1\nc()  # 2（count 跨调用保持）"),
            ("p", "经典坑：lambda 捕获的是循环变量 i 本身，不是创建那一刻的值——循环结束后所有 lambda 读到的都是最终的 i。修复：lambda i=i: i（用默认参数把当时的值固定下来）。"),
        ],
    },
}


def get_card(topic: str) -> Optional[Dict[str, object]]:
    return CARDS.get(topic)


def all_card_topics() -> List[str]:
    return list(CARDS)
