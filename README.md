# 알고리즘 공부 
교재 솔루션 : https://sites.math.rutgers.edu/~ajl213/CLRS/CLRS.html
### 알고리즘 개념
1. 알고리즘이란?
  > 주어진 입력을 출력으로 이끄는 잘 정의된 계산 절차   
  > 계산 문제를 정의하기 위해 입력과 출력의 관계를 잘 정의해야 하는데, 이러한 관계를 구현할 수 있는 계산과정  

2. 타당한 알고리즘
  > 알고리즘이 모든 입력 사례(Instance)에 대해 항상 올바른 출력을 내고, 종료할 경우를 타당하다고 한다  
  > 일부 입력 사례에 종료하지 않거나, 잘못된 답을 도출할 경우 타당하지 않다 일컺는다  

3. 어떤 문제를 알고리즘으로 풀어야할까?
  > 문제들은 공통적으로 "후보는 많지만, 대부분이 문제의 답이 아니다. 최상의 답을 찾는 것을 도전한다", "실용적인 사례를 주변에서 찾을 수 있다"와 같은 특징을 갖는다.
  
4. 알고리즘은 왜 필요할까?
  > 컴퓨터는 유한한 성능을 갖는다. 이로인해 공간적, 시간적 제약이 생긴다.  
  > 이러한 제약사항을 고려하여, 자원을 보다 효율적으로 사용해야 한다. = 효율적인 알고리즘이 필요하다.
  
5. 공부방법
  > 입력과 출력 그리고 사례(인스턴스)를 확실하게 이해한다  
  > 사례를 통해 아이디어(기본적인 이해)를 정리한다  
  > 아이디어를 유사코드로 만들고, 성능을 분석-파악한다  
  > 시간, 공간 복잡도를 이용하여 알고리즘을 효율적으로 최적화한다.  
<hr/>

### 삽입정렬
  - 입력 : n개 수들의 수열  (A Sequence of n numbers <a1, a2, ... , an>)
  - 출력 : a1 ≤ a2 ≤ a3 ≤ ... ≤ an을 만족하는 입력 수열의 순열(재배치)   (A Permutation <b1, b2, ... , bn> of the input sequence such that b1 ≤ b2 ≤ ... ≤ bn)
  - 유사코드
  ~~~
  for j = 2 to A.length
      key = A[j]
      i = j-1
      while i>0 and A[i] > key
          A[i+1] = A[i]
          i = i-1
      A[i+1] = key
  ~~~
  - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%EC%82%BD%EC%9E%85%EC%A0%95%EB%A0%AC.gif" width="500" height="300" />  
  
  - 성능 분석하기
  ~~~                                     cost          times
   for j = 2 to A.length                  c₁            bn
      key = A[j]                          c₂            n-1
      i = j-1                             c₃            n-1
      while i>0 and A[i] > key            c₄            Σtᴊ (j=2 ~ n)
          A[i+1] = A[i]                   c₅            Σ(tᴊ-1) (j=2 ~ n)
          i = i-1                         c₆            Σ(tᴊ-1) (j=2 ~ n)
      A[i+1] = key                        c₇            n-1
      
      *) tᴊ는 j가 2, 3, .., n 일때, while 루프의 검사가 실행되는 횟수를 의미
  ~~~
    > 수행시간은 각 명령문 수행시간의 합  
    > T(n) = c₁n + c₂(n-1) + c₃(n-1) + c₄Σtᴊ (j=2 ~ n) + c₅Σ(tᴊ-1) (j=2 ~ n) + c₆Σ(tᴊ-1) (j=2 ~ n) + c₇(n-1)  
    > (최악의 경우) T(n) = c₁n + c₂(n-1) + c₃(n-1) + c₄(n(n+1)/2-1) + c₅(n(n-1)/2)+ c₆(n(n-1)/2) + c₇(n-1)  
    > (최악의 경우) T(n) = an² + bn + c
 <hr/>
 
 
### 합병정렬
  - 입력 : n개 수들의 수열  (A Sequence of n numbers <a1, a2, ... , an>)
  - 출력 : a1 ≤ a2 ≤ a3 ≤ ... ≤ an을 만족하는 입력 수열의 순열(재배치)   (A Permutation <b1, b2, ... , bn> of the input sequence such that b1 ≤ b2 ≤ ... ≤ bn)
  - 분할(정렬할 n개 원소의 배열을 n/2개씩 부분 수열 두개로 분할),   
    정복(합병정렬을 통해 두 부분배열을 재귀적으로 정렬),   
    결합(정렬된 두 부분배열을 병합해 정렬된 배열 하나로 만든다) 세가지 포인트가 존재. 
  - 유사코드 (보조 프로시저인 "결합" 과정)
  ~~~
  Merge(A,p,q,r)  : A는 배열, p,q,r은 인덱스로 p <= q < r을 만족한다.                  cost           times
      n₁ = q-p+1                                                                      c₁             1
      n₂ = r-q                                                                        c₂             1
      배열 L[1 .. n₁+1]과 R[1 .. n₂+1]을 생성한다                                      c₃             1
      
      for i = 1 to n₁                                                                 c₄             n₁
          L[i] = A[p+i-1]                                                             c₅             1
      
      for j = 1 to n₂                                                                 c₆             n₂
          R[j] = A[q+j]                                                               c₇             1
      
      L[n₁+1] = ∞                                                                     c₈             1
      R[n₂+1] = ∞                                                                     c₉             1
      i = 1                                                                           c₁₀            1
      j = 1                                                   
      
      for k = p to r                                                                  c₁₁            n
          if L[i] <= R[j]                                                             c₁₂            c
              A[k] = L[i]                                                             c₁₃            1
              i = i+1                                                                 c₁₄            1
          else A[k] = R[j]                                                            c₁₅            1
              j = j+1                                                                 c₁₆            1
  ~~~
      > T(n) = c₄n₁ + c₆n₂ + c₁₁n + C
      > T(n) = an + b = Θ(n)
     
  
  - 유사코드 (합병정렬 전체 과정)
  ~~~                                                                                     
     Merge-Sort(A,p,r)
        if p < r
            q = ⌊(p+r)/2⌋
            Merge-Sort(A,p,q)
            Merge-Sort(A,q+1,r)
            Merge(A,p,q,r)
  ~~~
      # 분할정복 분석
      > 크기 n이 충분히 작아 n <= c 조건을 만족할 경우, T(n) = Θ(1)
      > 다른 모든 경우엔, T(n) = aT(n/b) + D(n) + C(n)
      *) 지금까지 a=b인 상황을 봤지만, 같지 않은 상황에서도 적용이 가능한 알고리즘도 분석할 예정  
      *) D(n)은 문제를 분할하는데 걸리는 시간, C(n) 부분 문제들의 해를 결합하여 원래 문제의 해를 만드는데 걸리는 시간  
      
      # 합병정렬 분석
      > 분할 : 부분 배열의 중간 위치를 계산하는 과정, 상수 시간이 걸린다. (D(n) = Θ(1))
      > 정복 : 두 개의 부분 문제를 재귀적으로 풀면서, 각 부분 문제는 크기가 n/2로 줄어든다. (2T(n/2))
      > 결합 : 앞서 보았듯, Merge 프로시저는 Θ(n) 시간이 걸린다. (C(n) = Θ(n))
      따라서, T(n) = 2T(n/2) + Θ(n) (n>1) / T(n) = Θ(1) (n=1)
      
      
  - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC.gif" width="500" height="300" />  
 <hr/>
 
### 알고리즘의 효율성
1. 점근적 분석
  > 알고리즘의 효율성을 판단할때, 주로 수행시간이 얼마나 걸리는지를 중심으로 판단한다.  
    이때, 각 알고리즘별 정확한 수행시간을 중심으로 판단하는 것이 아닌, 입력의 크기가 극한으로 증가할 때 수행시간을 중심으로 판단한다.
  > 이는 입력의 크기가 아주 작은 경우를 제외하고, 대부분의 경우에 좋은 판단기법이 되며 이러한 방식을 점근적 분석이라 한다.  
  > 이때, 입력의 크기가 극한으로 커가므로, 수행시간의 증가 차수 중 상수계수나 저차항은 무시된다.

2. 점근적 표기
  > 위에서 살펴본 점근적 분석을 형식적으로 표기하는 방법을 의미한다.
<hr/>

### 점근적 표기
- 알고리즘의 점근적 수행시간을 나타내는데 사용하는 표기.
- 정의역의 경우 자연수의 집합 N ={0,1,2 ...}로 정의된다.
  
  1. O 표기 (빅오 표기)  
    > 명칭 : 주어진 함수 g(n)에 대한 빅오g(n) 또는 오g(n)이라 명명한다. (O(g(n)))  
    > 정의 : O(g(n)) = { f(n) : 모든 n > n₀에 대해 0 <= f(n) <= cg(n)인 양의 상수 c, n₀이 존재한다. }  
           : f(n) = O(g(n)): g(n)는 f(n)에 대한 점근 상한을 의미한다.  
    > 함수의 상한을 나타내기 위해 사용되는 개념이다.  
      O는 알고리즘의 최악의 실행 시간을 설명하는데 좋다.  
    +) 추가이해  
    https://ko.wikipedia.org/wiki/%EC%A0%90%EA%B7%BC_%ED%91%9C%EA%B8%B0%EB%B2%95  
    
  2. Ω표기 (오메가 표기)  
    > 명칭 : 주어진 함수 g(n)에 대한 빅 오메가 g(n) 또는 오메가 g(n)이라 명명한다. (Ω(g(n))  
    > 정의 : Ω(g(n)) = { f(n) : 모든 n > n₀에 대해 0 <= cg(n) <= f(n) 인 양의 상수 c, n₀이 존재한다.}  
           : f(n) = O(g(n)): g(n)는 f(n)에 대한 점근 하한을 의미한다.  
    > 함수의 하한을 나타내기 위해 사용되는 개념이다.  
      Ω은 알고리즘의 최상의 사례 실행 시간을 설명하는데 좋다.  
      
  3. Θ표기 (세타 표기)  
    > 명칭 : 주어진 함수 g(n)에 대해 세타g(n)이라 명명한다. (Θ(g(n))  
    > 정의 : Θ(g(n)) = = { f(n) : 모든 n > n₀에 대해 0 <= c₁g(n) <= f(n) <= c₂g(n)인 양의 상수 c₁, c₂, n₀이 존재한다.}  
           : 
           
<hr/>

### T(n) 증명하기  
  - 수행시간에 대한 점화식은 다양한 형태로 표기된다. 이렇게 "추측"된 점화식을 어떻게 증명할 수 있을까?
  - "치환법(substitution method)", "재귀 트리 방법(recursion-tree method", "마스터 방법(master method)"등의 방법을 통해 증명한다.
  - 보통 등식의 꼴을 띄지만, 간혹 부등식 형태의 점화식도 사용된다.(부등식의 경우 빅오 또는 오메가 표기를 통해 사용한다.)

* 치환법
  -수학적 귀납법을 사용한다.
    1) 초항에 대한 증명을 보여준다.
    2) n = k-1까지의 모든 경우에서 해당 점화식이 참임을 가정한다.
    3) n = k일때 점화식이 참임을 증명한다.  
    ex)  
     ~~~
      T(n) = O(n*log n)을 증명하라
      1) 초항을 증명하라 / 정의역 n은 1보다 큰 자연수다. & T(1) = 1 & T(n) = 2T(n/2) + n
        n=2 > T(2)        <= 2*c*log₂2
              = 2T(1)+1   <= 2c
              = 4         <= 2c
              ∴ c >= 2이면 항상 참
      2) n = k-1까지 참이라고 가정
      3) n = k일때 참임을 증명하기
        T(k) = 2T(k/2) + k    > T(k/2)는 k-1까지 범위에 들어감. 즉 증명되어 있음.
                                ∴ T(k/2) <= c*(k/2)*log(k/2)
        2T(k/2) + k <= 2*(c*(k/2)*log(k/2)) + k
        T(k) <= c*k*logk - ck + k
        T(k) <= c*k*logk + (1-c)k
          > 1-c 는 항상 0보다 작은 수가 나온다. 
        ∴ T(k) <= c*k*logk + (1-c)k <= c*k*logk
           즉, T(k) <= c*k*logk (O(nlogn))을 만족한다.
     ~~~

* 재귀 트리 방법  
  1) 개념 이해하기
    - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC_%EC%9E%AC%EA%B7%80%ED%8A%B8%EB%A6%AC.png" width="500" height="300" /> 
    - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC_%EC%9E%AC%EA%B7%80%ED%8A%B8%EB%A6%AC2.png" width="500" height="300" /> 
    - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%ED%95%A9%EB%B3%91%EC%A0%95%EB%A0%AC_%EC%9E%AC%EA%B7%80%ED%8A%B8%EB%A6%AC3.png" width="500" height="300" />
  <br/><br/><br/>  

  2) 사례로 이해하기  
      ex)  
        ~~~
          case 1 : 줄어드는 크기가 동일
          T(n) = 2T(n/2)+n
          계층을 i로 두었을 때, 각 층의 노드 갯수는 2^i이다.
          또한 계층이 i일때, T(n)에서 n의 식은 n/2^i로 표현할 수 있다. 
          이때 가장 하층의 T(1)이 위치하게 되는데. 이는 즉 마지막 층 i 일때, n의 식이 1이 나온다를 의미한다. 이를 식으로 정리하면,
          n/2^i = 1 
          > n = 2^i
          > i = log₂n
            ∴ 마지막 층의 노드 갯수는 2^i 즉 2^log₂n = n이다.
            ∴ 각 계층별로의 합들은 해당 계층의 노드들을 다 더한 것과 같고, 전체 시간 T(n)은 모든 노드를 더한 것과 같다.
              = n + n + ... + n (개수는 높이인 i 즉 log₂n)
              = n * log₂n
              <= c * (n * log₂n)
              = O(n * log₂n)을 만족한다. 
        ~~~
        ~~~
          case 2 : 줄어드는 크기가 상이
          T(n) = T(n/3) + T(2n/3) + cn
                  
                      cn                               cn                                                                 cn
                T(n/3)  T(2n/3)    >        c(n/3)            c(2n/3)              >                            c(n/3)             c(2n/3)
                                       T(n/9)  T(2n/9)    T(2n/9)  T(4n/9)                                c(n/9)  c(2n/9)     c(2n/9)  c(4n/9)
                                                                                                                  .                   .
                                                                                                                  .                   .
                                                                                                   T(1) ---------------------------------------
                                                                                                        T(1) ------------------------------------
                                                                                                            T(1) ---------------------------------
                                                                                                              .....................................
                                                                                                                                                  T(1)
          위와 같이 방향에 따라 T(1)이 등장하는 높이가 상이하다. 
          (가장 크게 작아지는 경우가 가장 빠르게 T(1)에 도달, 반대로 작게 작아지는 경우 가장 느리게 T(1)에 도달)
          1) 이때 가장 빠르게 T(1)이 도달하는 방향의 높이를 k₁, 가장 느리게 T(1)이 도달하는 방향의 높이를 k₂이라고 했을때,
          트리에서 T(1) 등장하는 높이들 중에서 k₁의 높이가 가장 작고, k₂의 높이가 가장 크다.
          
          2) k₁과 k₂을 구하면, 
            n/3^k₁ = 1                 n*(2/3)^k₂ = 1
            3^k₁ = n                   n = (3/2)^k₂
            k₁ = log₃n                 k₂ = log(3/2)n
          
          3) ∴ k1 <= level <= k2
                = log₃n <= level <= log(3/2)n
          
          4) 각 레벨에서 노드의 합은 n,  T(n)은 모든 노트의 합임
             ∴ n*log₃n <= T(n) <= n*log(3/2)n
                = n*(log₂n/log₂3) <= T(n) <= n*(log₂n/log₂(3/2))
                = n*log₂n*c1 <= T(n) <= n*log₂n*c2
          
          5) ∴ T(n) <= c*n*log₂n 
                >T(n) = O(n*log₂n) 
        
        
        * 증명하기 / T(n) = T(n/3) + T(2n/3) + n
        1) base case 
            T(3) = T(1) + T(2) + 3 <= d * 3 * log₂3
            T(1), T(2)는 더 이상 분기할 수 없음(상수 시간 소모)
            ∴ 5 <= d * 3 * log₂3
               5 - 3*log₂3 <= d를 만족하는 d가 1개라도 존재... 증명됨
               
        2) n < k 일때 참이라고 가정
        3) n = k일 때 증명하기
            T(k) = T(k/3) + T(2k/3) + k    
            T(k/3), T(2k/3) 은 모두 n < k인 case. 따라서 전제에 의해 <= d*n*log₂n를 만족
            ∴ T(k/3) + T(2k/3) + k <= d*(1/3)*k*(log₂k - log₂3) + d*(2/3)*k*((1+log₂k) - log₂3) + k
                                    <= d*k*log₂k + k - d*k*(log₂3 - (2/3)) ... <= d*k*log₂k
            ∴ T(n) <= d*n*log₂n
            
            
       ~~~
        
<hr>
### HEAP

1. 힙이란?  
  가. 완전이진트리로 볼 수 있는 배열 객체
  나. 트리의 각 노드는 배열에서 한 원소와 대응된다.
  다. 가장 낮은 레벨을 빼고는 완전히 차 있고, 가장 낮은 레벨은 왼쪽부터 채운다.
  라. 트리의 루트는 [1]이며, 인덱스 i가 주어졌을 경우 부모의 인덱스는 ⌊i/2⌋ 왼쪽 자식의 인덱스는 2i, 오른쪽 자식의 인덱스는 2i+1이다
  마. 이진 힙에는 최대 힙과 최소 힙 두가지 종류가 존재한다. 종류에 따라 힙 특성이 다르고, 힙의 모든 노드들은 힙 특성을 만족한다. 이때 특성은 다음과 같다. "임의의 노드 값은 그 부모의 값보다 클(작을) 수 없다."
  바. 힙 트리에서 특정 노드의 높이는, 해당 노드에서 리프노드에 이르는 하향 경로 중 가장 긴 것의 간선 수로 정의 (힙의 높이는 루트 노드의 높이)
  ~~~
  
    힙의 노드 : O(lg n)
    높이가 k일때, 
    힙이 가질 수 있는 최소 노드 갯수는 (1 + 2 + 2^2 + 2^3 + ...  + 2^k-1 + 1) = 2^k
    힙이 가질 수 있는 최대 노드 갯수는 (1 + 2 + 2^2 + 2^3 + ... + 2^k) = 2^k+1 -1
    즉, 힙의 노드 개수 n은 다음과 같은 범위에 들어온다.
    2^k <= n <= 2^k+1 -1 < 2^k+1
    ∴ k <= lg n < k+1
       k = ⌊lg n⌋  즉, 노드 개수가 n개인 힙의 높이는 ⌊lg n⌋이다.
  ~~~

2. heap-size와 length   
  가. 배열의 인자들
  나. length는 배열이 가지는 원소의 개수를, heap-size는 배열의 원소 중 힙에 속하는 원소의 개수를 의미한다.
  다. 즉, 1 ~ length까지의 값들이 모두 유용할 순 있지만, 힙에 속하는 원소는 1 ~ heap-size 까지이다.  
      

3. MAX-HEAPIFY, BUILD-MAX-HEAP, HEAPSORT
  가. MAX-HEAPIFY 아이디어
  ~~~
    MAX-HEAPIFY(A,i)
      l = LEFT(i)
      r = RIGHT(i)
      if l <= A.heap-size and A[l] > A[i]
        largest = l
      else largest = i
      if r<= A.heap-size and A[r] > A[i]
        largest = r
      if largest != i
        exchange A[i] with A[largest]
        MAX-HEAPIFY(A, largest)
  ~~~
  나. MAX-HEAPIFY Worst Case
      (1) 왼쪽부터 노드가 차기 때문에, 항상 왼쪽의 노드 수가 오른쪽보다 많거나 같다.
      (2) 따라서 최악의 경우는, 왼쪽 노드를 타고 탐색하는 것이 최악의 경우이다.
      (3) 이떄, T(n) <= T(2n/3) + Ω(1)
  
  다. BUILD-MAX-HEAP 아이디어
  ~~~
    BUILD-MAX-HEAP(A)
      A.heap-size = A.length
      for i = ⌊A.length/2⌋ downto 1
        MAX-HEAPIFY(A,i)  
  ~~~
  
  라. BUILD-MAX-HEAP
<br/>
<hr/>
### 이진 검색트리

- 기본 설명
  (1) 트리
      하나의 부모의 다수의 자식들이 올 수 있다
  
  (2) 이진 트리
      각 노드마다 0~2의 자식을 갖는 트리
  
  (3) 검색 트리를 이루는 요소
      [1] 검색, 최소, 최대, 직전, 직후, 삽입, 삭제 등
      [2] 딕셔너리와 우선순위 큐를 이용
      [3] 기본 연산은 트리의 높이에 비례한다
  
  (4) 트리별 높이 시간 복잡도
      [1] 완전 이진 트리 : Ω(lg n)
      [2] 선형 체인 트리 : Ω(n)
      [3] 임의로 만들어진 이진 트리 : O(lg n)
  
  (5) 용어
      [1] 루트 : 조상이 없다
      [2] 리프 : 자식이 없다
      [3] 내부 : 리프가 아닌 노드
      [4] 높이 : 루트에서 리프까지의 거리
  
  (6) 이진 트리의 종류
      [1] Degenerate
          오직 한명의 자식을 갖는 트리
          선형 리스트와 유사
          높이 : O(n) (n개의 노드에서)
      [2] Balanced
          거의 두명의 자식을 갖는 트리
          검색에 유용한 트리
          높이 : O(lg n) (n개의 노드에서)
      [3] Complete
          항상 두명의 자식을 갖는 트리
       
       
- 이진 검색 트리의 개념, 특성
  (1) 이진 검색 트리의 개념
      key, 데이터, left(왼쪽 자식), right(오른쪽 자식), p(부모) 등의 필드를 갖는다.     
  (2) 이진 검색 트리의 특성
      x가 이진 검색 트리의 한 노드이다. 이때 y가 x의 왼쪽 서브 트리의 한 노드이면, y.key <= x.key를 만족한다. 그리고 y가 x의 오른쪽 서브 트리의 한 노드이면, y.key >= x.key를 만족한다. 
  (3) 구현 관점의 특성
      연결된 데이터 구조로 표현됨(각 노드는 객체를 의미)   > 키 + 위성 데이터

  
- 이진 검색 트리 순회
  > 트리의 각 노드에서 재귀적으로 호출하므로, n개의 노드로 이루어진 이진 검색 트리에서 Θ(n)을 만족
  
  (1) 전위 트리 순회
    ~~~
        PREORDER-TREE-WALK(x)
            if x != NIL
                print x.key
                PREORDER-TREE-WALK(x.left)           
                PREORDER-TREE-WALK(x.right)
    ~~~ 
        
    
    > 시간 복잡도
    ~~~
        1. BaseCase
        2. n이 k보다 작을때, 참이라고 가정
        3. n이 k일때, 증명
    ~~~
        >  
        
  (2) 중위 트리 순회
    ~~~
        INORDER-TREE-WALK(x)
            if x != NIL
                INORDER-TREE-WALK(x.left)           
                print x.key
                INORDER-TREE-WALK(x.right)
    ~~~ 
    
    > 시간 복잡도
    ~~~
        1. BaseCase
        2. n이 k보다 작을때, 참이라고 가정
        3. n이 k일때, 증명
    ~~~
        > n개의 노드로 이루어진 서브 트리의 루트에 대하여 호출시, 걸리는 시간을 T(n)이라 한다.
          중위 트리 순회는 n개의 노드를 모두 방문하기에, T(n) = Ω(n)
        > 빈 트리의 경우, 상수 c>0에 대해 T(0) = c (경미한 상수 시간 소모)         
        > n>0의 경우 왼쪽 서브 트리가 k개의 노드를, 오른쪽의 서브 트리가 n-k-1을 갖는다.
          이를 다시 표현하면 T(n) <= T(k) + T(n-k-1) + d
        > 치환을 통해, T(n) <= (c+d)n + c를 증명
          1) n = 0
            (c+d) * 0 + c = c = T(0)
          2) T(n) <= T(k) + T(n-k-1) + d
                  <= ((c+d)k + c) + ((c+d)(n-k-1) + c) + d
                  <= (c+d)n + c - (c+d) + c + d
                  <= (c+d)n + c
                  
    
  (3) 후위 트리 순회
    ~~~
        POSTORDER-TREE-WALK(x)
            if x != NIL
                POSTORDER-TREE-WALK(x.left)           
                POSTORDER-TREE-WALK(x.right)
                print x.key
    ~~~ 
    
    
    
- 이진 검색 트리 검색
    ~~~
        TREE-SEARCH(x,k)
            if x == NIL or K == x.key
                return x
            if k < x.key
                return TREE-SEARCH(x.left, k)
            else
                return TREE-SEARCH(x.right, k)
    ~~~ 
    > 시간 복잡도
    ~~~
        1. BaseCase
        2. n이 k보다 작을때, 참이라고 가정
        3. n이 k일때, 증명
    ~~~
        > 트리의 높이가 h일때, O(h)



- 이진 검색 트리 최소, 최대 원소
    (1) 최소 원소
    ~~~
        TREE-MINIMUM(x)
            while x.left != NIL
                x = x.left
            return x
    ~~~ 
    
    (2) 최대 원소
    ~~~
        TREE-MAXIMUM(x)
            while x.right != NIL
                x = x.right
            return x
    ~~~ 
     
     > 시간 복잡도
    ~~~
        1. BaseCase
        2. n이 k보다 작을때, 참이라고 가정
        3. n이 k일때, 증명
    ~~~
        > 트리의 높이가 h일때, O(h)
        > 이진 탐색 트리의 특성이 해당 알고리즘의 정확함을 보장한다. 이는 루트로부터 내려가는 하나의 단순 경로에 존재함을 보장



- 이진 검색 트리 직후, 직전 원소
    (1) 직후 원소
    ~~~
        TREE-SUCCESSOR(x)
            if x.right != NIL
                return TREE-MINIMUM(x.right)
            
            y = x.p
            
            while y != NIL and x == y.right
                x = y
                y = y.p
            
            return y
    ~~~ 
    
    (2) 직전 원소
    ~~~
        TREE-PREDECESSOR(x)
            if x.left != NIL
                return TREE-MAXIMUM(x.left)
                
            y = x.p
            
            while y != NIL and x == y.left
                x = y
                y = y.p
                
            return y
    ~~~ 
     
     > 시간 복잡도
    ~~~
        1. BaseCase
        2. n이 k보다 작을때, 참이라고 가정
        3. n이 k일때, 증명
    ~~~
        > 트리의 높이가 h일때, O(h)
        > 이진 탐색 트리의 특성이 해당 알고리즘의 정확함을 보장한다. 이는 루트로부터 내려가는 하나의 단순 경로에 존재함을 보장
        
        
        
- 이진 검색 트리 삽입
    ~~~
        TREE-INSERT(T,z)
            y = NIL
            x = T.root
            
            while x != NIL
                y = x
                if z.key < x.key
                    x = x.left
                else 
                    x = x.right
            
            z.p = y
            
            if y == NIL
                T.root = z
            else if z.key < y.key
                y.left = z
            else 
                y.right = z
    ~~~ 
    
     > 시간 복잡도
    ~~~
        1. BaseCase
        2. n이 k보다 작을때, 참이라고 가정
        3. n이 k일때, 증명
    ~~~
        > 트리의 높이가 h일때, O(h)
  




- 이진 검색 트리 삭제 & TRANSPLANT
    (1) TRANSPLANT
    ~~~
        TRANSPLANT(T,u,v)
            if u.p == NIL               -- if u doesn't have a parent => u is the root
                T.root = v              --   then v must replace u as the root of the tree T
             
            else if u == u.p.left       -- if u is a left subtree of its parent
                u.p.left = v            --   then v must replace u as the left
                                        --   subtree of u's parent
            else                        -- otherwise u is a right subtree 
                u.p.right = v           --   (as the tree is binary) and v must replace
                                        --   u as the right subtree of u's parent
            if v != NIL                 -- if v has replaced u (and thus is not NIL)
                v.p = u.p               --   v must have the same parent as u
    ~~~ 
        > 한 서브트리를 다른 서브 트리로 교체하는 루틴
       
    (2) TREE DELETE
    - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/binary-search-tree-deletion-algorithm.jpg" width="1000" height="300" />
    - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/delete-leaf-node-in-binary-search-tree.jpg" width="1000" height="300" />  
    - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/delete-node-in-binary-search-tree.jpg" width="1000" height="300" />    
    ~~~
        TREE-DELETE(T,z)
            if z.left == NIL
                TRANSPLANT(T,z,z.right)
            else if z.right == NIL
                TRANSPLANT(T,z,z.left)
            else
                y = TREE-MINIMUM(z.right)
                if y.p != NIL
                    TRANSPLANT(T,y,y.right)
                    y.right = z.right
                    y.right.p = y
                TRANSPLANT(T,z,y)
                y.left = z.left
                y.left.p = y
    ~~~ 
     
     > 시간 복잡도
    ~~~
        1. BaseCase
        2. n이 k보다 작을때, 참이라고 가정
        3. n이 k일때, 증명
    ~~~
        > 트리의 높이가 h일때, O(h)
        
<br/>
<br/>
<br/>
<hr/>

## AVL Trees
    
  - 기본 설명
      (1) 이진 탐색 트리의 성질을 기반으로 가짐
      (2) 트리 내의 어떤 노드에서든, 노드 기준 좌측 서브트리의 높이와 우측 서브트리의 높이 차이는 최대 1이다.

  - AVL 트리의 특성
      뿌리 왼쪽과 오른쪽 하위 트리의 높이가 최대 1씩 차이가 나고, 오른쪽 및 왼쪽 하위 트리도 AVL 트리인 BST이다.

  - AVL에서의 삽입
      (1) BST와 마찬가지로 각 노드별 높이 균형을 보장하기 위한 회전을 통한 운영
      (2) 4가지의 경우 존재
          [1] Single Left Rotation
          [2] Double Left Rotation
          [3] Single Right Rotation
          [4] Double Right Rotation
          > SLR, DLR : left(+2)
          > SRR, DRR : right(-2)
          - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/case1.png" width="1000" height="300" />
          - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/case2.png" width="1000" height="300" />
          - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/case3.png" width="1000" height="300" /> 
        
