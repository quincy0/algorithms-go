package daily

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
