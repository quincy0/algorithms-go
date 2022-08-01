package daily

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
	"unicode"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

/*
https://leetcode.cn/problems/complete-binary-tree-inserter/
Constructor: O(n)
*/
type CBTInserter struct {
	Root  *TreeNode
	Queue []*TreeNode
}

func Constructor(root *TreeNode) CBTInserter {
	temp := []*TreeNode{root}
	queue := []*TreeNode{}
	for len(temp) > 0 {
		node := temp[0]
		temp = temp[1:]
		if node.Left != nil {
			temp = append(temp, node.Left)
		}
		if node.Right != nil {
			temp = append(temp, node.Right)
		}
		if node.Left == nil || node.Right == nil {
			queue = append(queue, node)
		}
	}

	return CBTInserter{
		Root:  root,
		Queue: queue,
	}
}

func (this *CBTInserter) Insert(val int) int {
	newNode := &TreeNode{
		Val: val,
	}
	this.Queue = append(this.Queue, newNode)
	parent := this.Queue[0]
	if parent.Left == nil {
		parent.Left = newNode
	} else {
		parent.Right = newNode
		this.Queue = this.Queue[1:]
	}
	return parent.Val
}

func (this *CBTInserter) Get_root() *TreeNode {
	return this.Root
}

func towSum(nums []int, target int) []int {
	numMap := make(map[int]int)
	for k, v := range nums {
		temp := target - v
		if i, ok := numMap[temp]; ok {
			return []int{i, k}
		}
		numMap[v] = k
	}
	return nil
}

// https://leetcode.cn/problems/shift-2d-grid/
func shiftGrid(grid [][]int, k int) [][]int {
	m, n := len(grid), len(grid[0])
	ans := make([][]int, m)
	for i := range ans {
		ans[i] = make([]int, n)
	}

	for i, row := range grid {
		for j, v := range row {
			index := (i*n + j + k) % (m * n)
			ans[index/n][index%n] = v
		}
	}
	return ans
}

//https://leetcode.cn/problems/binary-tree-pruning/
func pruneTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left = pruneTree(root.Left)
	root.Right = pruneTree(root.Right)
	if root.Left == nil && root.Right == nil && root.Val == 0 {
		return nil
	}
	return root
}

// https://leetcode.cn/problems/add-two-numbers/
type ListNode struct {
	Val  int
	Next *ListNode
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var tail *ListNode
	var head *ListNode
	temp := 0
	for l1 != nil || l2 != nil {
		n1, n2 := 0, 0
		if l1 != nil {
			n1 = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			n2 = l2.Val
			l2 = l2.Next
		}
		sum := n1 + n2 + temp
		sum, temp = sum%10, sum/10
		if head == nil {
			head = &ListNode{Val: sum}
			tail = head
		} else {
			tail.Next = &ListNode{Val: sum}
			tail = tail.Next
		}
	}
	if temp > 0 {
		tail.Next = &ListNode{Val: temp}
	}
	return head
}

// https://leetcode.cn/problems/binary-search/
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		num := nums[mid]
		if num == target {
			return mid
		} else if num > target {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return -1
}

// https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/
func towSumSorted(numbers []int, target int) []int {
	left, right := 0, len(numbers)-1
	for left < right {
		sum := numbers[left] + numbers[right]
		if sum == target {
			return []int{left + 1, right + 1}
		} else if sum < target {
			left++
		} else {
			right--
		}
	}
	return nil
}

/**
1206. 设计跳表
https://leetcode.cn/problems/design-skiplist/
*/
const maxLevel = 32
const pFactor = 0.25

type SkiplistNode struct {
	val     int
	forward []*SkiplistNode
}

type Skiplist struct {
	head  *SkiplistNode
	level int
}

func SkipListConstructor() Skiplist {
	return Skiplist{
		head:  &SkiplistNode{val: -1, forward: make([]*SkiplistNode, maxLevel)},
		level: 0,
	}
}

func (Skiplist) randomLevel() int {
	lv := 1
	for lv < maxLevel && rand.Float64() < pFactor {
		lv++
	}
	return lv
}

func (this *Skiplist) Search(target int) bool {
	curr := this.head
	for i := this.level - 1; i >= 0; i-- {
		for curr.forward[i] != nil && curr.forward[i].val < target {
			curr = curr.forward[i]
		}
	}
	curr = curr.forward[0]
	return curr != nil && curr.val == target
}

func (this *Skiplist) Add(num int) {
	update := make([]*SkiplistNode, maxLevel)
	for i := range update {
		update[i] = this.head
	}
	curr := this.head
	for i := this.level - 1; i >= 0; i-- {
		for curr.forward[i] != nil && curr.forward[i].val < num {
			curr = curr.forward[i]
		}
		update[i] = curr
	}
	lv := this.randomLevel()
	this.level = max(lv, this.level)
	newNode := &SkiplistNode{
		val:     num,
		forward: make([]*SkiplistNode, lv),
	}
	for i, node := range update[:lv] {
		newNode.forward[i] = node.forward[i]
		node.forward[i] = newNode
	}
}

func (this *Skiplist) Erase(num int) bool {
	update := make([]*SkiplistNode, maxLevel)
	curr := this.head
	for i := this.level - 1; i >= 0; i-- {
		for curr.forward[i] != nil && curr.forward[i].val < num {
			curr = curr.forward[i]
		}
		update[i] = curr
	}
	curr = curr.forward[0]
	if curr == nil || curr.val != num {
		return false
	}
	for i := 0; i < this.level && update[i].forward[i] == curr; i++ {
		update[i].forward[i] = curr.forward[i]
	}
	for this.level > 1 && this.head.forward[this.level-1] == nil {
		this.level--
	}
	return true
}

func (this *Skiplist) Erasee(num int) bool {
	update := make([]*SkiplistNode, maxLevel)
	curr := this.head
	for i := this.level - 1; i >= 0; i-- {
		for curr.forward[i] != nil && curr.forward[i].val < num {
			curr = curr.forward[i]
		}
		update[i] = curr
	}
	curr = curr.forward[0]
	if curr == nil || curr.val != num {
		return false
	}
	for i := 0; i < this.level && update[i].forward[i] == curr; i++ {
		update[i].forward[i] = curr.forward[i]
	}

	for this.level > 1 && this.head.forward[this.level-1] == nil {
		this.level--
	}
	return true
}

func max(a, b int) int {
	if b > a {
		return b
	}
	return a
}

/**
https://leetcode.cn/problems/fraction-addition-and-subtraction/
时间复杂度：O(n+logC) n是expression长度，C是化简前结果分子分母最大值，求最大公约数需要O(logC)
空间复杂度：O(1)
*/
func fractionAddition(expression string) string {
	denominator, numerator := 0, 1
	for i, n := 0, len(expression); i < n; {
		denominator1, sign := 0, 1
		if expression[i] == '-' || expression[i] == '+' {
			if expression[i] == '-' {
				sign = -1
			}
			i++
		}
		for i < n && unicode.IsDigit(rune(expression[i])) {
			denominator1 = denominator1*10 + int(expression[i]-'0')
			i++
		}
		denominator1 = denominator1 * sign

		// "/"
		i++

		numerator1 := 0
		for i < n && unicode.IsDigit(rune(expression[i])) {
			numerator1 = numerator1*10 + int(expression[i]-'0')
			i++
		}
		denominator = denominator*numerator1 + denominator1*numerator
		numerator = numerator * numerator1
	}
	if denominator == 0 {
		return "0/1"
	}
	g := gcd(abs(denominator), numerator)
	return fmt.Sprintf("%d/%d", denominator/g, numerator/g)
}

func gcd(a, b int) int {
	for a != 0 {
		a, b = b%a, a
	}
	return b
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

/**
https://leetcode.cn/problems/rank-transform-of-an-array/

*/
func arrayRankTransform(arr []int) []int {
	sortedArr := append([]int{}, arr...)
	sort.Ints(sortedArr)
	m := make(map[int]int)
	for _, v := range sortedArr {
		if _, ok := m[v]; !ok {
			m[v] = len(m) + 1
		}
	}
	ans := make([]int, len(arr))
	for k, v := range arr {
		ans[k] = m[v]
	}
	return ans
}

/**
https://leetcode.cn/problems/maximum-level-sum-of-a-binary-tree/
时间复杂度：O(n)
空间复杂度：O(n)
*/
func maxLevelSum(root *TreeNode) int {
	ans, maxSum := 1, root.Val
	q := []*TreeNode{root}
	for level := 1; len(q) > 0; level++ {
		sum := 0
		p := q
		q = nil
		for _, node := range p {
			sum += node.Val
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
		}
		if sum > maxSum {
			ans = level
			maxSum = sum
		}
	}
	return ans
}

func generateTheString(n int) string {
	if n%2 == 1 {
		return strings.Repeat("a", n)
	}
	return strings.Repeat("a", n-1) + "b"
}
