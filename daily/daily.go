package daily

import (
	"fmt"
	"math/rand"
	"sort"
	"strconv"
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

/*
https://leetcode.cn/problems/orderly-queue/
时间复杂度：O(n^2)
空间复杂度：O(n)
*/
func orderlyQueue(s string, k int) string {
	if k == 1 {
		ans := s
		for i := 0; i < len(s); i++ {
			s = s[1:] + s[:1]
			if s < ans {
				ans = s
			}
		}
		return ans
	}
	t := []byte(s)
	sort.Slice(t, func(i, j int) bool {
		return t[i] < t[j]
	})
	return string(t)
}

/**
https://leetcode.cn/problems/minimum-subsequence-in-non-increasing-order/
时间复杂度：O(nlogn)
空间复杂度：O(logn)
*/
func minSubsequence(nums []int) []int {
	sort.Sort(sort.Reverse(sort.IntSlice(nums)))
	tot := 0
	for _, num := range nums {
		tot += num
	}
	for i, sum := 0, 0; ; i++ {
		sum += nums[i]
		if sum > tot-sum {
			return nums[:i+1]
		}
	}
}

/**
https://leetcode.cn/problems/exclusive-time-of-functions/
*/
func exclusiveTime(n int, logs []string) []int {
	ans := make([]int, n)
	type pair struct {
		idx, timestamp int
	}
	stack := []pair{}
	for _, log := range logs {
		data := strings.Split(log, ":")
		idx, _ := strconv.Atoi(data[0])
		timestamp, _ := strconv.Atoi(data[2])
		if data[1][0] == 's' {
			if len(stack) > 0 {
				ans[stack[len(stack)-1].idx] += timestamp - stack[len(stack)-1].timestamp
				stack[len(stack)-1].timestamp = timestamp
			}
			stack = append(stack, pair{idx, timestamp})
		} else {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			ans[idx] += timestamp - p.timestamp + 1
			if len(stack) > 0 {
				stack[len(stack)-1].timestamp = timestamp + 1
			}
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/special-binary-string/
*/
func makeLargestSpecial(s string) string {
	if len(s) <= 2 {
		return s
	}
	subs := sort.StringSlice{}
	cnt, left := 0, 0
	for i, ch := range s {
		if ch == '1' {
			cnt++
		} else if cnt--; cnt == 0 {
			subs = append(subs, "1"+makeLargestSpecial(s[left+1:i])+"0")
			left = i + 1
		}
	}
	sort.Sort(sort.Reverse(subs))
	return strings.Join(subs, "")
}

/**
https://leetcode.cn/problems/minimum-value-to-get-positive-step-by-step-sum/
*/
func minStartValue(nums []int) int {
	min, sum := 0, 0
	for _, v := range nums {
		sum += v
		if min > sum {
			min = sum
		}
	}
	return 1 - min
}

/**
https://leetcode.cn/problems/solve-the-equation/
时间复杂度：O(n)
空间复杂度：O(1)
*/
func SolveEquation(equation string) string {
	factor, val, sign := 0, 0, 1
	for i, n := 0, len(equation); i < n; {
		if equation[i] == '=' {
			sign = -1
			i++
			continue
		}
		s := sign
		if equation[i] == '+' {
			i++
		} else if equation[i] == '-' {
			s = -s
			i++
		}

		num, valid := 0, false
		for i < n && unicode.IsDigit(rune(equation[i])) {
			valid = true
			num = num*10 + int(equation[i]-'0')
			i++
		}
		if i < n && equation[i] == 'x' {
			if valid {
				s *= num
			}
			factor += s
			i++
		} else {
			val += s * num
		}

	}
	if factor == 0 {
		if val == 0 {
			return "Infinite solutions"
		}
		return "No solution"
	}
	return fmt.Sprintf("x=%d", -val/factor)
}

/**
https://leetcode.cn/problems/reformat-the-string/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func reformat(s string) string {
	digitCount := 0
	for _, v := range s {
		if unicode.IsDigit(v) {
			digitCount++
		}
	}
	alphaCount := len(s) - digitCount
	if abs(digitCount-alphaCount) > 1 {
		return ""
	}
	flag := digitCount > alphaCount
	t := []byte(s)
	for i, j := 0, 1; i < len(t); i += 2 {
		if unicode.IsDigit(rune(t[i])) != flag {
			for unicode.IsDigit(rune(t[j])) != flag {
				j += 2
			}
			t[i], t[j] = t[j], t[i]
		}
	}
	return string(t)
}

/**
https://leetcode.cn/problems/group-the-people-given-the-group-size-they-belong-to/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func groupThePeople(groupSizes []int) [][]int {
	groups := map[int][]int{}
	ans := [][]int{}
	for k, size := range groupSizes {
		groups[size] = append(groups[size], k)
	}
	for size, people := range groups {
		for i := 0; i < len(people); i += size {
			ans = append(ans, people[i:i+size])
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/max-chunks-to-make-sorted-ii/submissions/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func maxChunksToSortedII(arr []int) int {
	sortedArr := append([]int{}, arr...)
	sort.Ints(sortedArr)
	cnt := make(map[int]int)
	ans := 0
	for i, x := range arr {
		cnt[x]++
		if cnt[x] == 0 {
			delete(cnt, x)
		}
		y := sortedArr[i]
		cnt[y]--
		if cnt[sortedArr[i]] == 0 {
			delete(cnt, sortedArr[i])
		}
		if len(cnt) == 0 {
			ans++
		}
	}
	return ans
}

func maxChunksToSortedII2(arr []int) int {
	st := []int{}
	for _, v := range arr {
		if len(st) == 0 || st[len(st)-1] <= v {
			st = append(st, v)
		} else {
			max := st[len(st)-1]
			st = st[:len(st)-1]
			for len(st) > 0 && st[len(st)-1] > v {
				st = st[:len(st)-1]
			}
			st = append(st, max)
		}
	}
	return len(st)
}

/**
https://leetcode.cn/problems/max-chunks-to-make-sorted/
时间复杂度：O(N)
空间复杂度：O(1)
*/
func maxChunksToSortedI(arr []int) int {
	ans, max := 0, 0
	for i, v := range arr {
		if v > max {
			max = v
		}
		if max == i {
			ans++
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/maximum-score-after-splitting-a-string/
时间复杂度：O(N)
空间复杂度：O(1)
*/
func maxScore(s string) int {
	score := int('1'-s[0]) + strings.Count(s[1:], "1")
	ans := score
	for _, v := range s[1 : len(s)-1] {
		if v == '0' {
			score++
		} else {
			score--
		}
		if ans < score {
			ans = score
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/design-circular-deque/
*/
type MyCircularDeque struct {
	head    *CircularDequeNode
	tail    *CircularDequeNode
	maxLen  int
	currLen int
}

type CircularDequeNode struct {
	val    int
	next   *CircularDequeNode
	prefix *CircularDequeNode
}

func CircularDequeConstructor(k int) MyCircularDeque {
	return MyCircularDeque{maxLen: k}
}

func (this *MyCircularDeque) InsertFront(value int) bool {
	if this.IsFull() {
		return false
	}
	newNode := &CircularDequeNode{val: value}
	if this.currLen == 0 {
		this.head = newNode
		this.tail = newNode
	} else {
		newNode.next = this.head
		this.head.prefix = newNode
		this.head = newNode
	}
	this.currLen++
	return true
}

func (this *MyCircularDeque) InsertLast(value int) bool {
	if this.IsFull() {
		return false
	}
	newNode := &CircularDequeNode{val: value}
	if this.currLen == 0 {
		this.head = newNode
		this.tail = newNode
	} else {
		this.tail.next = newNode
		newNode.prefix = this.tail
		this.tail = newNode
	}
	this.currLen++
	return true
}

func (this *MyCircularDeque) DeleteFront() bool {
	if this.IsEmpty() {
		return false
	}
	this.head = this.head.next
	if this.head == nil {
		this.tail = nil
	}
	this.currLen--
	return true
}

func (this *MyCircularDeque) DeleteLast() bool {
	if this.IsEmpty() {
		return false
	}
	this.tail = this.tail.prefix
	if this.tail == nil {
		this.head = nil
	}
	this.currLen--
	return true
}

func (this *MyCircularDeque) GetFront() int {
	if this.currLen == 0 {
		return -1
	}
	return this.head.val
}

func (this *MyCircularDeque) GetRear() int {
	if this.currLen == 0 {
		return -1
	}
	return this.tail.val
}

func (this *MyCircularDeque) IsEmpty() bool {
	return this.currLen == 0
}

func (this *MyCircularDeque) IsFull() bool {
	return this.maxLen == this.currLen
}

/**
https://leetcode.cn/problems/design-an-ordered-stream/
时间复杂度：O(N)
空间复杂度：O(N)
*/
type OrderedStream struct {
	ptr  int
	data []string
}

func OrderedStreamConstructor(n int) OrderedStream {
	return OrderedStream{ptr: 1, data: make([]string, n)}
}

func (this *OrderedStream) Insert(idKey int, value string) []string {
	this.data[idKey-1] = value
	if this.ptr == idKey {
		i := idKey
		for i < len(this.data) && this.data[i] != "" {
			i++
		}
		this.ptr = i + 1
		return this.data[idKey-1 : i]
	}
	return []string{}
}

/**
https://leetcode.cn/problems/deepest-leaves-sum/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func deepestLeavesSum(root *TreeNode) (sum int) {
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		sum = 0
		size := len(queue)
		for i := 0; i < size; i++ {
			node := queue[0]
			queue = queue[1:]
			sum += node.Val
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
	}
	return
}

func deepestLeavesSumII(root *TreeNode) (sum int) {
	maxLevel := -1
	var dfs func(*TreeNode, int)
	dfs = func(node *TreeNode, level int) {
		if node == nil {
			return
		}
		if level > maxLevel {
			maxLevel = level
			sum = node.Val
		} else if level == maxLevel {
			sum += node.Val
		}
		dfs(node.Left, level+1)
		dfs(node.Right, level+1)
	}
	dfs(root, 0)
	return
}

/**
https://leetcode.cn/problems/maximum-equal-frequency/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func maxEqualFreq(nums []int) int {
	count, freq, maxFreq := make(map[int]int), make(map[int]int), 0
	ans := 0

	for i, num := range nums {
		if count[num] > 0 {
			freq[count[num]]--
		}
		count[num]++
		freq[count[num]]++
		if maxFreq < count[num] {
			maxFreq = count[num]
		}
		if maxFreq == 1 || freq[maxFreq]*maxFreq+freq[maxFreq-1]*(maxFreq-1) == i+1 && freq[maxFreq] == 1 || freq[maxFreq]*maxFreq+1 == i+1 && freq[1] == 1 {
			if ans < i+1 {
				ans = i + 1
			}
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/number-of-students-doing-homework-at-a-given-time/
时间复杂度：O(N)
空间复杂度：O(1)
*/
func busyStudent(startTime []int, endTime []int, queryTime int) int {
	count := 0
	for i := 0; i < len(startTime); i++ {
		if startTime[i] <= queryTime && endTime[i] >= queryTime {
			count++
		}
	}
	return count
}

/**
https://leetcode.cn/problems/maximum-binary-tree/
时间复杂度：O(N^2)
空间复杂度：O(N)
*/
func constructMaximumBinaryTreeI(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	index := 0
	for i, v := range nums {
		if v > nums[index] {
			index = i
		}
	}
	root := &TreeNode{
		Val:   nums[index],
		Left:  constructMaximumBinaryTreeI(nums[:index]),
		Right: constructMaximumBinaryTreeI(nums[index+1:]),
	}
	return root
}

/**
时间复杂度：O(N)
空间复杂度：O(N)
*/
func constructMaximumBinaryTreeII(nums []int) *TreeNode {
	tree := make([]*TreeNode, len(nums))
	stk := []int{}
	for i, num := range nums {
		tree[i] = &TreeNode{Val: num}

		for len(stk) > 0 && num > nums[stk[len(stk)-1]] {
			tree[i].Left = tree[stk[len(stk)-1]]
			stk = stk[:len(stk)-1]
		}
		if len(stk) > 0 {
			tree[stk[len(stk)-1]].Right = tree[i]
		}
		stk = append(stk, i)
	}
	return tree[stk[0]]
}
