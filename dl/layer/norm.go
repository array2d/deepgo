package layer

import "git.array2d.com/ai/deepgo/dl"

func NewNorm[T dl.Number](inFeatures, outFeatures int) (l *ComputeGraphNode[T]) {
	l = NewNode[T](1, 1)
	l.SetAttr("inFeatures", inFeatures)
	l.SetAttr("outFeatures", outFeatures)
	return
}
