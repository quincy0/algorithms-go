package tree

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

/**
https://leetcode.cn/problems/binary-tree-inorder-traversal/
时间复杂度：O(n)
空间复杂度：O(n)
*/
func inorderTraversalRecurse(root *TreeNode) []int {
	res := make([]int, 0)
	var inorder func(node *TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		res = append(res, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	return res
}

func inorderTraversalIterate(root *TreeNode) []int {
	stack := []*TreeNode{}
	ans := []int{}
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		ans = append(ans, root.Val)
		root = root.Right
	}
	return ans
}

/**
https://leetcode.cn/problems/binary-tree-preorder-traversal/
时间复杂度：O(n)
空间复杂度：O(n)
*/
func preorderTraversalRecurse(root *TreeNode) []int {
	ans := []int{}
	var preorder func(node *TreeNode)
	preorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		ans = append(ans, node.Val)
		preorder(node.Left)
		preorder(node.Right)
	}
	preorder(root)
	return ans
}
func preorderTraversalIterate(root *TreeNode) []int {
	stack := []*TreeNode{}
	ans := []int{}
	for root != nil || len(stack) > 0 {
		for root != nil {
			ans = append(ans, root.Val)
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1].Right
		stack = stack[:len(stack)-1]
	}
	return ans
}

/**
https://leetcode.cn/problems/binary-tree-postorder-traversal/
时间复杂度：O(n)
空间复杂度：O(n)
*/
func postorderTraversalRecurse(root *TreeNode) []int {
	ans := []int{}
	var postorder func(node *TreeNode)
	postorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		postorder(node.Left)
		postorder(node.Right)
		ans = append(ans, node.Val)
	}
	postorder(root)
	return ans
}

func postorderTraversalIterate(root *TreeNode) []int {
	ans := []int{}
	stack := []*TreeNode{}
	var prev *TreeNode
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if root.Right == nil || root.Right == prev {
			ans = append(ans, root.Val)
			prev = root
			root = nil
		} else {
			stack = append(stack, root)
			root = root.Right
		}
	}
	return ans
}

/**
https://leetcode.cn/problems/binary-tree-level-order-traversal/
时间复杂度：O(n)
空间复杂度：O(n)
*/
func levelOrder(root *TreeNode) [][]int {
	ans := [][]int{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		levelTotal := len(queue)
		temp := []int{}
		for i := 0; i < levelTotal; i++ {
			node := queue[0]
			queue = queue[1:]
			temp = append(temp, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		ans = append(ans, temp)
	}
	return ans
}

/**
https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

             1
         2       3
      4    5   6   7
    8   9        10  11
1 -> 2,3  -> 7,6,5,4 -> 8,9,10,11

*/
func zigzagLevelOrder(root *TreeNode) [][]int {
	ans := [][]int{}
	if root == nil {
		return ans
	}
	queue := []*TreeNode{root}
	for level := 0; len(queue) > 0; level++ {
		temp := []int{}
		q := queue
		queue = nil
		for _, node := range q {
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
			temp = append(temp, node.Val)
		}
		if level%2 == 1 {
			for i, n := 0, len(temp); i < n/2; i++ {
				temp[i], temp[n-i-1] = temp[n-i-1], temp[i]
			}
		}
		ans = append(ans, temp)
	}
	return ans
}
