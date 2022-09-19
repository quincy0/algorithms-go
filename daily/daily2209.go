package daily

import (
	"container/heap"
	"math"
	"sort"
	"strconv"
	"strings"
)

/**
https://leetcode.cn/problems/final-prices-with-a-special-discount-in-a-shop/
时间复杂度：O(N^2)
空间复杂度：O(1)
*/
func finalPrices(prices []int) []int {
	for i, p := range prices {
		for j := i + 1; j < len(prices); j++ {
			if p >= prices[j] {
				prices[i] = p - prices[j]
				break
			}
		}
	}
	return prices
}

func finalPricesII(prices []int) []int {
	stack := []int{0}
	ans := make([]int, len(prices))
	for i := len(prices) - 1; i >= 0; i-- {
		p := prices[i]
		for len(stack) > 1 && stack[len(stack)-1] > p {
			stack = stack[:len(stack)-1]
		}
		ans[i] = p - stack[len(stack)-1]
		stack = append(stack, p)
	}
	return ans
}

/**
https://leetcode.cn/problems/longest-univalue-path/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func longestUnivaluePath(root *TreeNode) int {
	ans := 0
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left, right := dfs(node.Left), dfs(node.Right)
		nodeLeft, nodeRight := 0, 0
		if node.Left != nil && node.Val == node.Left.Val {
			nodeLeft = 1 + left
		}
		if node.Right != nil && node.Val == node.Right.Val {
			nodeRight = 1 + right
		}
		if nodeLeft+nodeRight > ans {
			ans = nodeLeft + nodeRight
		}
		if nodeLeft > nodeRight {
			return nodeLeft
		}
		return nodeRight
	}
	dfs(root)
	return ans
}

/**
https://leetcode.cn/problems/maximum-length-of-pair-chain/
时间复杂度：O(NlogN)
空间复杂度：O(logN)
*/
func findLongestChain(pairs [][]int) int {
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i][1] < pairs[j][1]
	})
	ans := 1
	a := pairs[0][1]
	for i := 1; i < len(pairs); i++ {
		b := pairs[i][0]
		if b > a {
			ans++
			a = pairs[i][1]
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/special-positions-in-a-binary-matrix/
时间复杂度：O(M*N)
空间复杂度：O(1)
*/
func numSpecial(mat [][]int) int {
	for i, row := range mat {
		cnt1 := 0
		for _, v := range row {
			if v == 1 {
				cnt1++
			}
		}
		if i == 0 {
			cnt1--
		}
		if cnt1 > 0 {
			for j, num := range row {
				if num == 1 {
					mat[0][j] += cnt1
				}
			}
		}
	}
	ans := 0
	for _, v := range mat[0] {
		if v == 1 {
			ans++
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/find-duplicate-subtrees/
时间复杂度：O(N) A B A AB BA ABA
空间复杂度：O(N)
*/
func findDuplicateSubtrees(root *TreeNode) []*TreeNode {
	type pair struct {
		node *TreeNode
		idx  int
	}
	seen := map[[3]int]pair{}
	repeat := map[*TreeNode]struct{}{}
	idx := 0
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		tri := [3]int{node.Val, dfs(node.Left), dfs(node.Right)}
		if p, ok := seen[tri]; ok {
			repeat[p.node] = struct{}{}
			return p.idx
		}
		idx++
		seen[tri] = pair{node, idx}
		return idx
	}
	dfs(root)
	ans := make([]*TreeNode, 0, len(repeat))
	for node := range repeat {
		ans = append(ans, node)
	}
	return ans
}

/**
https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func UniqueLetterString(s string) int {
	idx := map[rune][]int{}
	ans := 0
	for i, c := range s {
		idx[c] = append(idx[c], i)
	}
	for _, arr := range idx {
		arr = append(append([]int{-1}, arr...), len(s))
		for i := 1; i < len(arr)-1; i++ {
			ans += (arr[i] - arr[i-1]) * (arr[i+1] - arr[i])
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/rearrange-spaces-between-words/
时间复杂度：O(N)
空间复杂度：O(N)
*/
func reorderSpaces(text string) string {
	words := strings.Fields(text)
	space := strings.Count(text, " ")
	l := len(words) - 1
	if l == 0 {
		return words[0] + strings.Repeat(" ", space)
	}
	return strings.Join(words, strings.Repeat(" ", space/l)) + strings.Repeat(" ", space%l)
}

/**
https://leetcode.cn/problems/beautiful-arrangement-ii/
1 3 2 4 n4 k2
*/
func constructArray(n, k int) []int {
	ans := make([]int, 0, n)
	for i := 1; i < n-k; i++ {
		ans = append(ans, i)
	}
	for i, j := n-k, n; i <= j; i++ {
		ans = append(ans, i)
		if i != j {
			ans = append(ans, j)
		}
		j--
	}
	return ans
}

/**
https://leetcode.cn/problems/crawler-log-folder/
*/
func minOperations(logs []string) int {
	deep := 0
	for _, s := range logs {
		if strings.HasPrefix(s, "../") {
			if deep > 0 {
				deep--
			}
		} else if strings.HasPrefix(s, "./") {
			continue
		} else {
			deep++
		}
	}
	return deep
}

/**
https://leetcode.cn/problems/trim-a-binary-search-tree
*/
func trimBST(root *TreeNode, low int, high int) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val < low {
		return trimBST(root.Right, low, high)
	}
	if root.Val > high {
		return trimBST(root.Left, low, high)
	}
	root.Left = trimBST(root.Left, low, high)
	root.Right = trimBST(root.Right, low, high)
	return root
}

func trimBSTII(root *TreeNode, low, high int) *TreeNode {
	for root != nil && (root.Val < low || root.Val > high) {
		if root.Val < low {
			root = root.Right
		} else {
			root = root.Left
		}
	}
	if root == nil {
		return nil
	}
	for node := root; node.Left != nil; {
		if node.Left.Val < low {
			node.Left = node.Left.Right
		} else {
			node = node.Left
		}
	}
	for node := root; node.Right != nil; {
		if node.Right.Val > high {
			node.Right = node.Right.Left
		} else {
			node = node.Right
		}
	}
	return root
}

func mincostToHireWorkers(quality, wage []int, k int) float64 {
	n := len(quality)
	h := make([]int, n)
	for i := range h {
		h[i] = i
	}
	sort.Slice(h, func(i, j int) bool {
		a, b := h[i], h[j]
		return quality[a]*wage[b] > quality[b]*wage[a]
	})
	totalq := 0
	q := hp{}
	for i := 0; i < k-1; i++ {
		totalq += quality[h[i]]
		heap.Push(&q, quality[h[i]])
	}
	ans := 1e9
	for i := k - 1; i < n; i++ {
		idx := h[i]
		totalq += quality[idx]
		heap.Push(&q, quality[idx])
		ans = math.Min(ans, float64(wage[idx])/float64(quality[idx])*float64(totalq))
		totalq -= heap.Pop(&q).(int)
	}
	return ans
}

type hp struct{ sort.IntSlice }

func (h hp) Less(i, j int) bool  { return h.IntSlice[i] > h.IntSlice[j] }
func (h *hp) Push(v interface{}) { h.IntSlice = append(h.IntSlice, v.(int)) }
func (h *hp) Pop() interface{} {
	a := h.IntSlice
	v := a[len(a)-1]
	h.IntSlice = a[:len(a)-1]
	return v
}

/**
https://leetcode.cn/problems/special-array-with-x-elements-greater-than-or-equal-x/
时间复杂度：O(NlogN)
空间复杂度：O(logN)
*/
func specialArray(nums []int) int {
	sort.Sort(sort.Reverse(sort.IntSlice(nums)))
	for i := 0; i < len(nums); i++ {
		if nums[i] >= i+1 && (i+1 == len(nums) || nums[i+1] < i+1) {
			return i + 1
		}
	}
	return -1
}

/**
https://leetcode.cn/problems/maximum-swap/submissions/
时间复杂度：O(logNum)
空间复杂度：O(logNum)
*/
func maximumSwap(num int) int {
	s := []byte(strconv.Itoa(num))
	n := len(s)
	maxId, id1, id2 := n-1, -1, -1
	for i := n - 1; i >= 0; i-- {
		if s[maxId] < s[i] {
			maxId = i
		} else if s[maxId] > s[i] {
			id1, id2 = i, maxId
		}
	}
	if id1 < 0 {
		return num
	}
	s[id1], s[id2] = s[id2], s[id1]
	ans, _ := strconv.Atoi(string(s))
	return ans
}

/**
https://leetcode.cn/problems/mean-of-array-after-removing-some-elements/
*/
func trimMean(arr []int) float64 {
	sort.Ints(arr)
	n := len(arr)
	sum := 0
	for _, num := range arr[n*5/100 : n*95/100] {
		sum += num
	}
	return float64(sum*10) / float64(n*9)
}

/**
https://leetcode.cn/problems/bulb-switcher-ii
时间复杂度：O(1)
空间复杂度：O(1)
*/
func flipLights(n int, presses int) int {
	//不按开关
	if presses == 0 {
		return 1
	}
	if n == 1 {
		return 2
	}
	if n == 2 {
		if presses == 1 {
			return 3
		}
		return 4
	}
	//n >= 3
	if presses == 1 {
		return 4
	} else if presses == 2 {
		return 7
	}
	return 8
}

type seg []struct{ cover, len, maxLen int }

func (t seg) init(hBound []int, idx, l, r int) {
	if l == r {
		t[idx].maxLen = hBound[l] - hBound[l-1]
		return
	}
	mid := (l + r) / 2
	t.init(hBound, idx*2, l, mid)
	t.init(hBound, idx*2+1, mid+1, r)
	t[idx].maxLen = t[idx*2].maxLen + t[idx*2+1].maxLen
}

func (t seg) update(idx, l, r, ul, ur, diff int) {
	if l > ur || r < ul {
		return
	}
	if ul <= l && r <= ur {
		t[idx].cover += diff
		t.pushUp(idx, l, r)
		return
	}
	mid := (l + r) / 2
	t.update(idx*2, l, mid, ul, ur, diff)
	t.update(idx*2+1, mid+1, r, ul, ur, diff)
	t.pushUp(idx, l, r)
}

func (t seg) pushUp(idx, l, r int) {
	if t[idx].cover > 0 {
		t[idx].len = t[idx].maxLen
	} else if l == r {
		t[idx].len = 0
	} else {
		t[idx].len = t[idx*2].len + t[idx*2+1].len
	}
}

/**
https://leetcode.cn/problems/rectangle-area-ii
*/
func rectangleArea(rectangles [][]int) (ans int) {
	n := len(rectangles) * 2
	hBound := make([]int, 0, n)
	for _, r := range rectangles {
		hBound = append(hBound, r[1], r[3])
	}
	// 排序，方便下面去重
	sort.Ints(hBound)
	m := 0
	for _, b := range hBound[1:] {
		if hBound[m] != b {
			m++
			hBound[m] = b
		}
	}
	hBound = hBound[:m+1]
	t := make(seg, m*4)
	t.init(hBound, 1, 1, m)

	type tuple struct{ x, i, d int }
	sweep := make([]tuple, 0, n)
	for i, r := range rectangles {
		sweep = append(sweep, tuple{r[0], i, 1}, tuple{r[2], i, -1})
	}
	sort.Slice(sweep, func(i, j int) bool { return sweep[i].x < sweep[j].x })

	for i := 0; i < n; i++ {
		j := i
		for j+1 < n && sweep[j+1].x == sweep[i].x {
			j++
		}
		if j+1 == n {
			break
		}
		// 一次性地处理掉一批横坐标相同的左右边界
		for k := i; k <= j; k++ {
			idx, diff := sweep[k].i, sweep[k].d
			// 使用二分查找得到完整覆盖的线段的编号范围
			left := sort.SearchInts(hBound, rectangles[idx][1]) + 1
			right := sort.SearchInts(hBound, rectangles[idx][3])
			t.update(1, 1, m, left, right, diff)
		}
		ans += t[1].len * (sweep[j+1].x - sweep[j].x)
		i = j
	}
	return ans % (1e9 + 7)
}

/**
https://leetcode.cn/problems/largest-substring-between-two-equal-characters
*/
func maxLengthBetweenEqualCharacters(s string) int {
	ans := -1
	firstIndex := [26]int{}
	for i := 0; i < 26; i++ {
		firstIndex[i] = -1
	}
	for i, c := range s {
		c -= 'a'
		if firstIndex[c] < 0 {
			firstIndex[c] = i
		} else {
			if i-firstIndex[c]-1 > ans {
				ans = i - firstIndex[c] - 1
			}
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/making-a-large-island/
*/
func largestIsland(grid [][]int) (ans int) {
	dir4 := []struct{ x, y int }{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	n, t := len(grid), 0
	tag := make([][]int, n)
	for i := range tag {
		tag[i] = make([]int, n)
	}
	area := map[int]int{}
	var dfs func(int, int)
	dfs = func(i, j int) {
		tag[i][j] = t
		area[t]++
		for _, d := range dir4 {
			x, y := i+d.x, j+d.y
			if 0 <= x && x < n && 0 <= y && y < n && grid[x][y] > 0 && tag[x][y] == 0 {
				dfs(x, y)
			}
		}
	}
	for i, row := range grid {
		for j, x := range row {
			if x > 0 && tag[i][j] == 0 { // 枚举没有访问过的陆地
				t = i*n + j + 1
				dfs(i, j)
				ans = max(ans, area[t])
			}
		}
	}

	for i, row := range grid {
		for j, x := range row {
			if x == 0 { // 枚举可以添加陆地的位置
				newArea := 1
				conn := map[int]bool{0: true}
				for _, d := range dir4 {
					x, y := i+d.x, j+d.y
					if 0 <= x && x < n && 0 <= y && y < n && !conn[tag[x][y]] {
						newArea += area[tag[x][y]]
						conn[tag[x][y]] = true
					}
				}
				ans = max(ans, newArea)
			}
		}
	}
	return
}

/**
https://leetcode.cn/problems/sort-array-by-increasing-frequency/
*/
func frequencySort(nums []int) []int {
	cnt := map[int]int{}
	for _, x := range nums {
		cnt[x]++
	}
	sort.Slice(nums, func(i, j int) bool {
		a, b := nums[i], nums[j]
		return cnt[a] < cnt[b] || cnt[a] == cnt[b] && a > b
	})
	return nums
}
