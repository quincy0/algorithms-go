package daily

import (
	"container/heap"
	"math"
	"sort"
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
