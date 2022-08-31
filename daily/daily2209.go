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
