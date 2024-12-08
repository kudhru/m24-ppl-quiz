(* Quiz Questions on OCaml let and in expressions *)

(* Question 1 *)
let q1 = {|
What is the value of x after executing the following code?

let x = 10 in
let x = let x = x * 2 in x + 5 in
let x = x - 3 in
x
|}

let a1 = "The answer is 22. Here's why:
1. Initial x = 10
2. Inner let: x = 10 * 2 = 20, then x + 5 = 25
3. Final let: x = 25 - 3 = 22"

(* Question 2 *)
let q2 = {|
What is the output of this nested let expression?

let x = 5 in
let y = let x = x + 3 in x * 2 in
let x = y + x in
x
|}

let a2 = "The answer is 21. Here's the evaluation:
1. Initial x = 5
2. In y's definition: x = 5 + 3 = 8, then 8 * 2 = 16 (y = 16)
3. Final x = 16 + 5 = 21"

(* Question 3 *)
let q3 = {|
What does this expression evaluate to?

let f = let x = 3 in fun y -> x + y in
let x = 10 in
f x
|}

let a3 = "The answer is 13. This demonstrates lexical scoping:
1. f captures x = 3 in its closure
2. When f is called with x = 10, it uses the captured x (3)
3. Result is 3 + 10 = 13"

(* Question 4 *)
let q4 = {|
What is the final value of result?

let x = 1 in
let result = 
  let x = x + 1 in
  let x = x * 3 in
  let x = let x = x + 2 in x * 2 in
  x + 5
in
result
|}

let a4 = "The answer is 17. Let's break it down:
1. Initial x = 1
2. x = 1 + 1 = 2
3. x = 2 * 3 = 6
4. Inner let: x = 6 + 2 = 8, then 8 * 2 = 16
5. Final result: 16 + 5 = 21"

(* Question 5 *)
let q5 = {|
What is the value of z?

let x = 5 in
let y = 
  let x = x * 2 in
  let z = x + 3 in
  z
in
let z = y + (let x = 2 in x + y) in
z
|}

let a5 = "The answer is 31. Here's the evaluation:
1. Initial x = 5
2. In y's definition: x = 5 * 2 = 10, z = 10 + 3 = 13, so y = 13
3. In z's definition: 
   - y = 13
   - New x = 2, so x + y = 2 + 13 = 15
4. Final z = 13 + 15 = 28"

(* Questions about = operator *)

let q6 = {|
What is the result of this comparison?

let x = [1; 2; 3] in
let y = 1 :: 2 :: [3] in
let z = [1; 2; 3] in
(x = y) && (y = z) && (physical_equal x z)
|}

let a6 = "The answer is false. Here's why:
1. x = y is true (structural equality of lists)
2. y = z is true (structural equality of lists)
3. physical_equal x z is false (different list objects in memory)
4. true && true && false = false"

let q7 = {|
What does this expression evaluate to?

type point = { x: float; y: float }
let p1 = { x = 1.0; y = 2.0 }
let p2 = { x = 1.0; y = 2.0 }
let p3 = { x = 1.0 +. 0.0; y = 2.0 }
(p1 = p2) && (p2 = p3)
|}

let a7 = "The answer is true. This demonstrates that:
1. = performs structural comparison of records
2. Float equality is used for float fields
3. 1.0 and 1.0 +. 0.0 are equal floats
4. All three records are structurally equal"

let q8 = {|
What is the result of this comparison?

let f1 = fun x -> x + 1
let f2 = fun x -> x + 1
let f3 = f1

(f1 = f2, f1 = f3)
|}

let a8 = "The answer is (false, true). Because:
1. Functions with same behavior but defined separately are not equal (f1 = f2 is false)
2. Function references to the same function are equal (f1 = f3 is true)
3. = on functions compares function identity, not behavior"

let q9 = {|
What does this evaluate to?

type shape = Circle of float | Rectangle of float * float

let s1 = Circle 2.0
let s2 = Circle (1.0 +. 1.0)
let s3 = Rectangle (2.0, 2.0)
let s4 = Rectangle (2.0, 2.0)

[(s1 = s2); (s3 = s4); (s1 = s3)]
|}

let a9 = "The answer is [true; true; false]. Here's why:
1. s1 = s2: true (same constructor, equal float values)
2. s3 = s4: true (same constructor, equal float pairs)
3. s1 = s3: false (different constructors)
The = operator compares variant constructors and their arguments"

let q10 = {|
What is the result?

let x = ref 1
let y = ref 1
let z = x

let result = [
  !x = !y;      (* Comparison 1 *)
  x = y;        (* Comparison 2 *)
  x = z;        (* Comparison 3 *)
  !x = !(ref 1) (* Comparison 4 *)
]
|}

let a10 = "The answer is [true; false; true; true]. Let's analyze each comparison:
1. !x = !y: true (comparing values: 1 = 1)
2. x = y: false (comparing different reference locations)
3. x = z: true (comparing same reference location)
4. !x = !(ref 1): true (comparing dereferenced values: 1 = 1)
This shows the difference between comparing reference values vs reference locations"

(* Questions about if-else expressions *)

let q11 = {|
What is the type and value of this expression?

let f x y = 
  if x > 0 then 
    if y > 0 then "both"
    else if x > y then "x only"
  else if y > 0 then "y only"

f 2 (-1)
|}

let a11 = "Type: string option, Value: Some \"x only\"
Explanation:
1. This is a partial function due to missing else branches
2. OCaml infers option type due to possible missing returns
3. x = 2 > 0, then y = -1 ≤ 0, then 2 > -1 is true"

let q12 = {|
What does this evaluate to?

let x = ref 5 in
let y = ref 10 in
let result = if (x := !x + 1; !x > 5) 
             then (y := !y + 1; !y) 
             else !x in
(!x, !y, result)
|}

let a12 = "The answer is (6, 11, 11). Here's why:
1. x is incremented to 6 in the condition
2. 6 > 5 is true, so the then branch executes
3. y is incremented to 11
4. result gets !y (11)
5. Final state: x=6, y=11, result=11"

let q13 = {|
What's the result and why?

let div_safe x y =
  if y = 0 then None
  else Some (x / y)

let result =
  if let Some q = div_safe 10 2 in q > 4 then
    "big"
  else if let None = div_safe 10 0 in true then
    "zero"
  else
    "small"
|}

let a13 = "The answer is \"small\". Let's analyze:
1. div_safe 10 2 returns Some 5
2. First pattern match succeeds, but 5 > 4 is false
3. Second if: pattern match with None succeeds
4. But the entire expression is in if condition, not the body
5. Falls through to else branch"

let q14 = {|
What's the type and value?

let f = function
  | x when if x > 0 then x mod 2 = 0 else x mod 2 = 1 ->
      if x >= 0 then `Positive else `Negative
  | _ -> `Zero

[(f 4); (f (-3)); (f 0); (f 3)]
|}

let a14 = "Type: [`Negative | `Positive | `Zero] list
Value: [`Positive; `Zero; `Zero; `Zero]
Because:
1. For x=4: x>0 true, 4 mod 2 = 0 true → `Positive
2. For x=-3: x>0 false, -3 mod 2 = -1 false → `Zero
3. For x=0: first guard fails → `Zero
4. For x=3: x>0 true, 3 mod 2 = 1 false → `Zero"

let q15 = {|
What does this evaluate to?

let rec f n =
  if n <= 0 then 1
  else if n mod 2 = 0 then 
    let x = f (n-1) in
    if x > 10 then x-1 else x+1
  else f (n-2)

f 5
|}

let a15 = "The answer is 3. Trace the recursion:
1. f 5 → f 3 → f 1 → f (-1) → 1
2. f 1 → f (-1) → 1
3. f 3 → f 1 → 1
4. f 5 → f 3 → 1
Shows how nested if-else affects recursion paths"

let q16 = {|
What's the result of this expression?

let x = ref None in
let y = ref 0 in
if (x := Some 5; y := !y + 1; !x = None) then
  !y
else if (y := !y * 2; !x = Some 5) then
  !y
else
  0
|}

let a16 = "The answer is 2. Here's the evaluation:
1. x is set to Some 5
2. y is incremented to 1
3. !x = None is false
4. y is multiplied by 2 (becomes 2)
5. !x = Some 5 is true
6. Returns !y which is 2"

let q17 = {|
What's wrong with this code and what's its type?

let f x y =
  if x > y then
    if x > 0 then x
    else if y < 0 then y
  else if y > x then y
  else x

type result = OK of int | Error
let safe_f x y = try OK (f x y) with _ -> Error
|}

let a17 = "Type: int -> int -> int option
Issues:
1. Missing else branches make it partial
2. Not all paths return same type
3. Some paths have no return value
4. OCaml infers option type due to partiality
The safe_f wrapper would handle the partiality"

let q18 = {|
What does this evaluate to and why?

let g x = if x then 1 else 0
let h x = if x = 0 then true else false

List.init 4 (fun i -> 
  if h (g (h (g (h i)))) 
  then 1 
  else 0)
|}

let a18 = "The answer is [1; 0; 0; 0]. Let's trace i=0:
1. h 0 = true
2. g true = 1
3. h 1 = false
4. g false = 0
5. h 0 = true
Therefore 1. Similar logic for i=1,2,3 gives 0"

let q19 = {|
What's the result and type?

let f = function
  | x when if x < 0 then true else x = 0 -> `Zero
  | x when if x > 0 then x < 10 else false -> `Small
  | _ -> `Big

[f (-1); f 0; f 5; f 10]
|}

let a19 = "Type: [`Big | `Small | `Zero] list
Value: [`Zero; `Zero; `Small; `Big]
Analysis:
1. For -1: x < 0 true → `Zero
2. For 0: x < 0 false, x = 0 true → `Zero
3. For 5: first guard false, second guard true → `Small
4. For 10: x > 0 true but x < 10 false → `Big"

let q20 = {|
What's the result?

let r = ref 0
let f x = r := !r + 1; !r > x
let g x = r := !r * 2; !r < x

let test x y =
  if f x && g y || f y && g x then
    !r
  else
    -(!r)

test 2 10
|}

let a20 = "The answer is 4. Short-circuit evaluation is key:
1. f 2 sets r to 1, 1 > 2 is false
2. f 10 sets r to 2, 2 > 10 is false
3. Due to short-circuit, g is never called
4. Returns -(!r) which is -2
Shows how if-else interacts with side effects and short-circuit evaluation"

(* Questions about Anonymous Functions *)

let q21 = {|
What does this evaluate to?

let compose f g = fun x -> f (g x)
let add1 = fun x -> x + 1
let double = fun x -> x * 2

let funcs = List.init 3 (fun i -> 
  compose (fun x -> x + i) (fun x -> x * i))

List.map (fun f -> f 5) funcs
|}

let a21 = "The answer is [0; 6; 17]. Here's why:
1. When i=0: (fun x -> x + 0) ∘ (fun x -> x * 0) → 5 * 0 + 0 = 0
2. When i=1: (fun x -> x + 1) ∘ (fun x -> x * 1) → 5 * 1 + 1 = 6
3. When i=2: (fun x -> x + 2) ∘ (fun x -> x * 2) → 5 * 2 + 2 = 17
Shows closure capture in composed anonymous functions"

let q22 = {|
What's the type and value?

let f = fun x -> fun y -> fun z -> 
  if x > y then (fun w -> w + z) 
  else (fun w -> w * z)

[(f 3 2 4) 5; (f 1 2 4) 5]
|}

let a22 = "Type: int list, Value: [9; 20]
Explanation:
1. f 3 2 4 returns (fun w -> w + 4) since 3 > 2, so (f 3 2 4) 5 = 9
2. f 1 2 4 returns (fun w -> w * 4) since 1 ≤ 2, so (f 1 2 4) 5 = 20
Shows how anonymous functions can return different functions"

let q23 = {|
What's the result?

let apply_n f n =
  let rec helper acc = function
    | 0 -> acc
    | n -> helper (f acc) (n-1)
  in helper

let g = apply_n (fun x -> x * 2) 3 
let h = apply_n (fun x -> x + 3) 2

[g 1; h 1]
|}

let a23 = "The answer is [8; 7]. Let's trace:
1. g 1 applies (fun x -> x * 2) three times:
   1 → 2 → 4 → 8
2. h 1 applies (fun x -> x + 3) twice:
   1 → 4 → 7
Shows composition of anonymous function application"

let q24 = {|
What does this evaluate to?

let funcs = [
  (fun x -> x + 1);
  (fun y -> y * 2);
  (fun z -> z * z)
]

let apply_all x = 
  List.fold_left (fun acc f -> f acc) x funcs

[apply_all 2; apply_all 3]
|}

let a24 = "The answer is [25; 49]. Here's the evaluation:
1. apply_all 2:
   2 → 3 → 6 → 36
2. apply_all 3:
   3 → 4 → 8 → 64
Shows chaining of anonymous functions through fold_left"

let q25 = {|
What's the result and why?

let memoize f =
  let cache = ref [] in
  fun x ->
    try List.assoc x !cache
    with Not_found ->
      let result = f x in
      cache := (x, result) :: !cache;
      result

let expensive = memoize (fun x -> x * x * x)
[expensive 3; expensive 3; expensive 2]
|}

let a25 = "The answer is [27; 27; 8]. Analysis:
1. First call computes 3³ = 27 and caches it
2. Second call retrieves 27 from cache
3. Third call computes 2³ = 8 and caches it
Shows how anonymous functions can be used in memoization"

let q26 = {|
What's the output?

let curry f = fun x -> fun y -> f (x, y)
let uncurry f = fun (x, y) -> f x y

let f = curry (fun (x, y) -> x + y)
let g = uncurry (fun x -> fun y -> x * y)

[(f 3 4); (g (2, 5))]
|}

let a26 = "The answer is [7; 10]. Here's why:
1. f 3 4 = (curry (fun (x, y) -> x + y)) 3 4 = 3 + 4 = 7
2. g (2, 5) = (uncurry (fun x -> fun y -> x * y)) (2, 5) = 2 * 5 = 10
Shows currying/uncurrying with anonymous functions"

let q27 = {|
What's the result?

let rec map_reduce mapper reducer base = function
  | [] -> base
  | x::xs -> reducer (mapper x) (map_reduce mapper reducer base xs)

let result = map_reduce 
  (fun x -> fun y -> x + y) 
  (fun f g -> fun x -> f (g x)) 
  (fun x -> x) 
  [fun x -> x+1; fun x -> x*2; fun x -> x-1]

result 3
|}

let a27 = "The answer is 8. Function composition order:
1. (x-1) → 2
2. (x*2) → 4
3. (x+1) → 5
Shows complex composition of anonymous functions with map-reduce"

let q28 = {|
What's the type and result?

let flip f = fun x y -> f y x
let compose f g = fun x -> f (g x)
let apply x f = f x

let f = flip compose (fun x -> x + 1)
let g = f (fun x -> x * 2)
let h = apply 3 g

h
|}

let a28 = "Type: int -> int, Result: 7
Explanation:
1. flip compose reverses composition order
2. f = fun g x -> (fun x -> x + 1) (g x)
3. g = fun x -> (fun x -> x + 1) ((fun x -> x * 2) x)
4. h = g 3 = (3 * 2) + 1 = 7
Shows function composition with flipped arguments"

let q29 = {|
What does this evaluate to?

let rec fix f x = f (fix f) x

let factorial = fix (fun self n ->
  if n <= 1 then 1
  else n * self (n-1))

let fibonacci = fix (fun self n ->
  if n <= 1 then n
  else self (n-1) + self (n-2))

[factorial 4; fibonacci 5]
|}

let a29 = "The answer is [24; 5]. Analysis:
1. factorial 4 = 4 * 3 * 2 * 1 = 24
2. fibonacci 5 = fib(4) + fib(3) = 3 + 2 = 5
Shows Y-combinator style recursion with anonymous functions"

let q30 = {|
What's the result?

let pipeline = 
  let ( |> ) x f = f x in
  let ( >> ) f g = fun x -> g (f x) in
  
  let ops = [
    (fun x -> x + 1) >> (fun x -> x * 2);
    (fun x -> x - 3) >> (fun x -> x * x);
    fun x -> x mod 5
  ]
  in
  
  List.fold_left (fun acc f -> f acc) 5 ops

pipeline
|}

let a30 = "The answer is 4. Let's trace:
1. First composition: (5 + 1) * 2 = 12
2. Second composition: (12 - 3)² = 81
3. Final function: 81 mod 5 = 1
Shows function composition with custom operators and fold"

(* Questions about Mutually Recursive Functions *)

let q31 = {|
What does this evaluate to?

type expr = 
  | Num of int
  | Add of expr * term
and term = 
  | One of int
  | Mul of term * expr

let rec eval_expr = function
  | Num n -> n
  | Add (e, t) -> eval_expr e + eval_term t
and eval_term = function
  | One n -> n
  | Mul (t, e) -> eval_term t * eval_expr e

let exp = Add(Num 3, Mul(One 2, Add(Num 1, One 4)))
eval_expr exp
|}

let a31 = "The answer is 13. Let's evaluate:
1. eval_expr (Add(Num 3, Mul(One 2, Add(Num 1, One 4))))
2. 3 + eval_term (Mul(One 2, Add(Num 1, One 4)))
3. 3 + (2 * eval_expr (Add(Num 1, One 4)))
4. 3 + (2 * (1 + 4))
5. 3 + (2 * 5) = 13
Shows mutually recursive evaluation of expression trees"

let q32 = {|
What's the result?

let rec is_even n = 
  if n = 0 then true
  else if n = 1 then false
  else is_odd (n-1)
and is_odd n =
  if n = 0 then false
  else if n = 1 then true
  else is_even (n-1)

let count_even_odd nums =
  List.fold_left (fun (e, o) n ->
    if is_even n then (e+1, o) else (e, o+1)
  ) (0, 0) nums

count_even_odd [5; 8; 12; 15; 21; 24; 30]
|}

let a32 = "The answer is (4, 3). Analysis:
1. 5: odd → (0, 1)
2. 8: even → (1, 1)
3. 12: even → (2, 1)
4. 15: odd → (2, 2)
5. 21: odd → (2, 3)
6. 24: even → (3, 3)
7. 30: even → (4, 3)
Shows mutual recursion for parity checking"

let q33 = {|
What's the output?

type tree = 
  | Leaf of int 
  | Node of forest
and forest = 
  | Empty 
  | Trees of tree * forest

let rec sum_tree = function
  | Leaf n -> n
  | Node f -> sum_forest f
and sum_forest = function
  | Empty -> 0
  | Trees (t, f) -> sum_tree t + sum_forest f

let t = Node(Trees(Leaf 1, 
            Trees(Node(Trees(Leaf 2, Empty)), 
            Trees(Leaf 3, Empty))))
sum_tree t
|}

let a33 = "The answer is 6. Let's trace:
1. Node contains forest Trees(Leaf 1, Trees(Node(...), Trees(Leaf 3, Empty)))
2. sum_forest evaluates:
   - sum_tree (Leaf 1) = 1
   - sum_tree (Node(Trees(Leaf 2, Empty))) = 2
   - sum_tree (Leaf 3) = 3
3. Total: 1 + 2 + 3 = 6
Shows mutual recursion with complex tree/forest structure"

let q34 = {|
What does this evaluate to?

let rec interleave xs ys = match (xs, ys) with
  | ([], _) -> ys
  | (_, []) -> xs
  | (x::xs', y::ys') -> x :: y :: (interleave xs' ys')

let rec split = function
  | [] -> ([], [])
  | [x] -> ([x], [])
  | x::y::rest -> 
      let (xs, ys) = split rest in
      (x::xs, y::ys)

let rec merge_sort = function
  | [] | [_] as l -> l
  | lst ->
      let (l1, l2) = split lst in
      interleave (merge_sort l1) (merge_sort l2)

merge_sort [3; 1; 4; 1; 5; 9; 2; 6]
|}

let a34 = "The answer is [1; 1; 2; 3; 4; 5; 6; 9]. Here's why:
1. split creates ([3;4;5;2], [1;1;9;6])
2. Recursively splits and interleaves
3. interleave maintains relative order while merging
4. Result is not fully sorted because interleave doesn't compare elements
Shows mutual recursion in list processing"

let q35 = {|
What's the result?

type expr =
  | Val of int
  | Add of expr * expr
  | Mul of expr * expr

let rec optimize = function
  | Add(e1, e2) -> optimize_add (optimize e1) (optimize e2)
  | Mul(e1, e2) -> optimize_mul (optimize e1) (optimize e2)
  | e -> e
and optimize_add e1 e2 = match (e1, e2) with
  | (Val 0, e) | (e, Val 0) -> e
  | (Val x, Val y) -> Val(x + y)
  | _ -> Add(e1, e2)
and optimize_mul e1 e2 = match (e1, e2) with
  | (Val 1, e) | (e, Val 1) -> e
  | (Val 0, _) | (_, Val 0) -> Val 0
  | (Val x, Val y) -> Val(x * y)
  | _ -> Mul(e1, e2)

let exp = Mul(Add(Val 0, Val 2), Add(Val 1, Val 3))
optimize exp
|}

let a35 = "The answer is Val 8. Optimization steps:
1. Add(Val 0, Val 2) → Val 2 (optimize_add)
2. Add(Val 1, Val 3) → Val 4 (optimize_add)
3. Mul(Val 2, Val 4) → Val 8 (optimize_mul)
Shows mutually recursive expression optimization"

let q36 = {|
What's the output?

type json =
  | JNull
  | JBool of bool
  | JNumber of float
  | JString of string
  | JArray of json list
  | JObject of (string * json) list

let rec validate_json schema value = match (schema, value) with
  | ("null", JNull) -> true
  | ("boolean", JBool _) -> true
  | ("number", JNumber _) -> true
  | ("string", JString _) -> true
  | ("array", JArray arr) -> List.for_all (validate_json "any") arr
  | ("object", JObject obj) -> List.for_all validate_object obj
  | ("any", _) -> true
  | _ -> false
and validate_object (_, value) = validate_json "any" value

let obj = JObject [
  ("name", JString "John");
  ("age", JNumber 30.0);
  ("data", JArray [JNumber 1.0; JNull])
]

[validate_json "object" obj;
 validate_json "array" obj;
 validate_json "any" obj]
|}

let a36 = "The answer is [true; false; true]. Because:
1. validate_json \"object\" obj: true (valid object structure)
2. validate_json \"array\" obj: false (not an array)
3. validate_json \"any\" obj: true (any accepts all values)
Shows mutual recursion in JSON validation"

let q37 = {|
What does this evaluate to?

type 'a stream = Cons of 'a * (unit -> 'a stream)

let rec interleave_streams s1 s2 = match s1 with
  | Cons(x, f1) -> Cons(x, fun () -> 
      interleave_streams s2 (f1 ()))

let rec sieve = function
  | Cons(p, f) -> Cons(p, fun () ->
      sieve (filter (f ()) p))
and filter stream p = match stream with
  | Cons(x, f) ->
      if x mod p = 0 then filter (f ()) p
      else Cons(x, fun () -> filter (f ()) p)

let rec nums_from n = Cons(n, fun () -> nums_from (n+1))
let rec take n s = match (n, s) with
  | (0, _) -> []
  | (n, Cons(x, f)) -> x :: take (n-1) (f ())

take 5 (sieve (nums_from 2))
|}

let a37 = "The answer is [2; 3; 5; 7; 11]. Here's why:
1. sieve starts with 2 and filters its multiples
2. filter removes multiples of each prime
3. Remaining numbers are prime
4. take 5 gets first 5 primes
Shows mutual recursion with infinite streams"

let q38 = {|
What's the result?

type expr = 
  | Int of int 
  | Add of expr * expr
  | Mul of expr * expr

let rec size = function
  | Int _ -> 1
  | Add(e1, e2) | Mul(e1, e2) -> size e1 + size e2

let rec build n = 
  if n <= 0 then Int 1
  else add_or_mul (build (n/2)) (build (n-1-n/2))
and add_or_mul e1 e2 =
  if size e1 > size e2 then Add(e1, e2)
  else Mul(e1, e2)

let rec eval = function
  | Int n -> n
  | Add(e1, e2) -> eval e1 + eval e2
  | Mul(e1, e2) -> eval e1 * eval e2

[size (build 4); eval (build 4)]
|}

let a38 = "The answer is [7; 24]. Analysis:
1. build 4 creates a tree with 7 nodes
2. Evaluation follows the structure:
   - Combines smaller expressions based on size
   - Results in specific evaluation order
Shows mutual recursion in expression building and evaluation"

let q39 = {|
What does this evaluate to?

type term =
  | Var of string
  | App of term * term
  | Abs of string * term

let rec free_vars = function
  | Var x -> [x]
  | App(t1, t2) -> 
      List.sort_uniq compare (free_vars t1 @ free_vars t2)
  | Abs(x, t) -> 
      List.filter ((<>) x) (free_vars t)

let rec subst x v = function
  | Var y when y = x -> v
  | Var y -> Var y
  | App(t1, t2) -> App(subst x v t1, subst x v t2)
  | Abs(y, t) when y = x -> Abs(y, t)
  | Abs(y, t) when not (List.mem y (free_vars v)) ->
      Abs(y, subst x v t)
  | Abs(y, t) -> Abs(y, t)

let t = App(Abs("x", App(Var "x", Var "y")), Var "z")
free_vars (subst "y" (Var "w") t)
|}

let a39 = "The answer is [\"w\"; \"z\"]. Here's why:
1. Original term: (λx.xy)z
2. Substituting w for y: (λx.xw)z
3. Free variables are w and z
Shows mutual recursion in lambda calculus operations"

let q40 = {|
What's the result?

type nat = Zero | Succ of nat

let rec nat_to_int = function
  | Zero -> 0
  | Succ n -> 1 + nat_to_int n

let rec add m n = match m with
  | Zero -> n
  | Succ m' -> Succ (add m' n)

let rec mul m n = match m with
  | Zero -> Zero
  | Succ m' -> add n (mul m' n)

let rec pow m n = match n with
  | Zero -> Succ Zero
  | Succ n' -> mul m (pow m n')

let two = Succ (Succ Zero)
let three = Succ two

nat_to_int (pow two three)
|}

let a40 = "The answer is 8. Let's evaluate:
1. pow two three = mul two (mul two two)
2. mul two two = add two two = Succ (Succ (Succ (Succ Zero)))
3. Final multiplication results in 2³ = 8
Shows mutual recursion with Peano numbers"

(* Easy Questions about Recursive Variants, Functions and Pattern Matching *)

let q61 = {|
What does this evaluate to?

type color = Red | Blue | Green

let describe_color = function
  | Red -> "warm"
  | Blue -> "cool"
  | Green -> "natural"

[describe_color Red; describe_color Blue]
|}

let a61 = "The answer is [\"warm\"; \"cool\"]. Simple pattern matching on variants:
1. describe_color Red matches first pattern → \"warm\"
2. describe_color Blue matches second pattern → \"cool\""

let q62 = {|
What's the result?

type shape = Circle | Square | Triangle

let rec count_circles = function
  | [] -> 0
  | Circle :: rest -> 1 + count_circles rest
  | _ :: rest -> count_circles rest

count_circles [Circle; Square; Circle; Triangle]
|}

let a62 = "The answer is 2. Let's count:
1. First Circle: count = 1
2. Skip Square
3. Second Circle: count = 2
4. Skip Triangle
Shows basic recursive pattern matching on lists"

let q63 = {|
What does this evaluate to?

type 'a option = None | Some of 'a

let safe_divide x y =
  if y = 0 then None
  else Some (x / y)

[safe_divide 10 2; safe_divide 10 0]
|}

let a63 = "The answer is [Some 5; None]. Here's why:
1. 10/2 = 5, so Some 5
2. Division by 0 returns None
Shows simple option type usage"

let q64 = {|
What's the result?

type intTree = 
  | Leaf 
  | Node of int * intTree * intTree

let rec sum_tree = function
  | Leaf -> 0
  | Node(value, left, right) -> value + sum_tree left + sum_tree right

let t = Node(1, Node(2, Leaf, Leaf), Node(3, Leaf, Leaf))
sum_tree t
|}

let a64 = "The answer is 6. Let's add:
1. Root value: 1
2. Left node: 2
3. Right node: 3
4. Total: 1 + 2 + 3 = 6
Shows basic tree traversal with recursion"

let q65 = {|
What does this evaluate to?

let rec length = function
  | [] -> 0
  | _::xs -> 1 + length xs

let rec drop n lst = match (n, lst) with
  | (0, xs) -> xs
  | (_, []) -> []
  | (n, _::xs) -> drop (n-1) xs

length (drop 2 [1; 2; 3; 4; 5])
|}

let a65 = "The answer is 3. Here's the process:
1. drop 2 [1;2;3;4;5] removes first two elements → [3;4;5]
2. length [3;4;5] counts remaining elements → 3
Shows combination of recursive functions"

let q66 = {|
What's the result?

type animal = Dog | Cat | Bird

let make_sound = function
  | Dog -> "woof"
  | Cat -> "meow"
  | Bird -> "tweet"

let animals = [Dog; Cat; Bird; Dog]
List.map make_sound animals
|}

let a66 = "The answer is [\"woof\"; \"meow\"; \"tweet\"; \"woof\"]. 
Shows simple pattern matching with List.map:
1. Each animal is matched to its sound
2. List.map applies make_sound to each element"

let q67 = {|
What does this evaluate to?

type weather = Sunny | Rainy | Cloudy

let rec count_sunny = function
  | [] -> 0
  | Sunny :: rest -> 1 + count_sunny rest
  | _ :: rest -> count_sunny rest

let rec count_rainy = function
  | [] -> 0
  | Rainy :: rest -> 1 + count_rainy rest
  | _ :: rest -> count_rainy rest

let weather_week = [Sunny; Rainy; Sunny; Cloudy; Rainy]
(count_sunny weather_week, count_rainy weather_week)
|}

let a67 = "The answer is (2, 2). Count results:
1. Sunny days: 2 (positions 1 and 3)
2. Rainy days: 2 (positions 2 and 5)
Shows multiple recursive functions on same list"

let q68 = {|
What's the result?

type grade = A | B | C | D | F

let is_passing = function
  | F -> false
  | _ -> true

let grades = [A; C; F; B; F]
List.filter is_passing grades
|}

let a68 = "The answer is [A; C; B]. Analysis:
1. is_passing returns false only for F
2. List.filter keeps elements where is_passing returns true
3. Both F grades are removed
Shows simple pattern matching with List.filter"

let q69 = {|
What does this evaluate to?

type 'a mylist = Empty | Cons of 'a * 'a mylist

let rec to_list = function
  | Empty -> []
  | Cons(x, rest) -> x :: to_list rest

let l = Cons(1, Cons(2, Cons(3, Empty)))
to_list l
|}

let a69 = "The answer is [1; 2; 3]. Conversion process:
1. Cons(1, ...) → 1 :: rest
2. Cons(2, ...) → 2 :: rest
3. Cons(3, Empty) → 3 :: []
Shows basic recursive data type conversion"

let q70 = {|
What's the result?

type expr = 
  | Number of int
  | Sum of expr * expr

let rec eval = function
  | Number n -> n
  | Sum(e1, e2) -> eval e1 + eval e2

let e = Sum(Number 3, Sum(Number 4, Number 5))
eval e
|}

let a70 = "The answer is 12. Evaluation steps:
1. Sum(Number 3, Sum(Number 4, Number 5))
2. 3 + eval(Sum(Number 4, Number 5))
3. 3 + (4 + 5)
4. 3 + 9 = 12
Shows simple expression evaluation with recursion"

(* Medium-level Questions about Lists and Pattern Matching *)

let q71 = {|
What does this evaluate to?

let rec alternate = function
  | [] -> []
  | [x] -> [x]
  | x :: y :: rest -> x :: (alternate rest)

alternate [1; 2; 3; 4; 5]
|}

let a71 = "The answer is [1; 3; 5]. Here's why:
1. First pattern: x=1, y=2, rest=[3;4;5]
2. 1 :: alternate [3;4;5]
3. 1 :: (3 :: alternate [5])
4. 1 :: 3 :: [5]
Shows skipping alternate elements with pattern matching"

let q72 = {|
What's the result?

let rec zip = function
  | ([], _) -> []
  | (_, []) -> []
  | (x::xs, y::ys) -> (x, y) :: zip (xs, ys)

zip ([1; 2; 3], ["a"; "b"; "c"; "d"])
|}

let a72 = "The answer is [(1,\"a\"); (2,\"b\"); (3,\"c\")]. Trace:
1. (1,\"a\") :: zip ([2;3], [\"b\";\"c\";\"d\"])
2. (1,\"a\") :: (2,\"b\") :: zip ([3], [\"c\";\"d\"])
3. (1,\"a\") :: (2,\"b\") :: (3,\"c\") :: []
Shows pairing elements until shorter list ends"

let q73 = {|
What does this evaluate to?

let rec split = function
  | [] -> ([], [])
  | [x] -> ([x], [])
  | x :: y :: rest ->
      let (xs, ys) = split rest in
      (x :: xs, y :: ys)

split [1; 2; 3; 4; 5]
|}

let a73 = "The answer is ([1; 3; 5], [2; 4]). Analysis:
1. Base case reached with [5]: ([5], [])
2. For 3,4: (3::xs, 4::ys) where (xs,ys) = ([5], [])
3. For 1,2: (1::xs, 2::ys) where (xs,ys) = ([3;5], [4])
Shows splitting list into odd and even positions"

let q74 = {|
What's the result?

let rec remove_duplicates = function
  | [] -> []
  | x :: xs ->
      let rec contains y = function
        | [] -> false
        | z :: zs -> z = y || contains y zs
      in
      if contains x (remove_duplicates xs)
      then remove_duplicates xs
      else x :: remove_duplicates xs

remove_duplicates [1; 2; 2; 3; 1; 4; 2]
|}

let a74 = "The answer is [4; 3; 1; 2]. Here's why:
1. Processes list from right to left
2. Each element is checked against already processed unique elements
3. Only first occurrence (from right) is kept
Shows duplicate removal with nested recursion"

let q75 = {|
What does this evaluate to?

let rec insert x = function
  | [] -> [x]
  | y :: ys as l -> if x <= y then x :: l else y :: insert x ys

let rec insertion_sort = function
  | [] -> []
  | x :: xs -> insert x (insertion_sort xs)

insertion_sort [3; 1; 4; 1; 5; 9; 2; 6]
|}

let a75 = "The answer is [1; 1; 2; 3; 4; 5; 6; 9]. Trace:
1. First sorts [1;4;1;5;9;2;6]
2. Then inserts 3 in correct position
3. Each element is inserted into already sorted portion
Shows insertion sort with pattern matching"

let q76 = {|
What's the result?

let rec group = function
  | [] -> []
  | x :: xs ->
      let rec take_while p = function
        | y :: ys when p y -> y :: take_while p ys
        | _ -> []
      in
      let same = x :: take_while ((=) x) xs in
      let rest = match xs with
        | y :: ys when y = x -> 
            let rec drop_while p = function
              | y :: ys when p y -> drop_while p ys
              | l -> l
            in drop_while ((=) x) xs
        | l -> l
      in
      same :: group rest

group [1; 1; 2; 3; 3; 3; 4; 4; 1]
|}

let a76 = "The answer is [[1; 1]; [2]; [3; 3; 3]; [4; 4]; [1]]. Process:
1. Groups consecutive equal elements
2. First finds all 1s, then moves to 2
3. Groups three 3s, then two 4s
4. Final single 1 forms its own group
Shows grouping consecutive elements"

let q77 = {|
What does this evaluate to?

let rec interleave xs ys = match (xs, ys) with
  | ([], l) -> l
  | (l, []) -> l
  | (x::xs', y::ys') -> x :: y :: interleave xs' ys'

let rec split = function
  | [] -> ([], [])
  | [x] -> ([x], [])
  | x :: y :: rest ->
      let (xs, ys) = split rest in
      (x :: xs, y :: ys)

let rec riffle lst =
  let (xs, ys) = split lst in
  interleave xs ys

riffle [1; 2; 3; 4; 5; 6]
|}

let a77 = "The answer is [1; 4; 2; 5; 3; 6]. Steps:
1. split creates ([1;2;3], [4;5;6])
2. interleave combines alternating elements
3. Result preserves relative order within each half
Shows list shuffling with split and merge"

let q78 = {|
What's the result?

let rec rotate_left n lst =
  if n <= 0 then lst else
  match lst with
  | [] -> []
  | x :: xs -> rotate_left (n-1) (xs @ [x])

let rec rotate_right n lst =
  if n <= 0 then lst else
  match lst with
  | [] -> []
  | _ ->
      let rec last_and_rest = function
        | [] -> (None, [])
        | [x] -> (Some x, [])
        | x :: xs ->
            let (last, rest) = last_and_rest xs in
            (last, x :: rest)
      in
      match last_and_rest lst with
      | (Some last, rest) -> rotate_right (n-1) (last :: rest)
      | _ -> []

[rotate_left 2 [1; 2; 3; 4; 5];
 rotate_right 2 [1; 2; 3; 4; 5]]
|}

let a78 = "The answer is [[3; 4; 5; 1; 2]; [4; 5; 1; 2; 3]]. Analysis:
1. rotate_left 2 moves first 2 elements to end
2. rotate_right 2 moves last 2 elements to front
Shows list rotation in both directions"

let q79 = {|
What does this evaluate to?

let rec sublists = function
  | [] -> [[]]
  | x :: xs ->
      let subs = sublists xs in
      let rec add_to_all x = function
        | [] -> []
        | l :: ls -> (x :: l) :: add_to_all x ls
      in
      subs @ add_to_all x subs

sublists [1; 2; 3]
|}

let a79 = "The answer is [[];[3];[2];[2;3];[1];[1;3];[1;2];[1;2;3]]. Here's why:
1. First gets sublists of [2;3]
2. Then adds 1 to each sublist
3. Combines original sublists with new ones
Shows generating all possible sublists"

let q80 = {|
What's the result?

let rec permutations = function
  | [] -> [[]]
  | lst ->
      let rec insert_all x = function
        | [] -> [[x]]
        | y :: ys as l ->
            (x :: l) :: 
            (match insert_all x ys with
             | [] -> []
             | z :: zs -> (y :: z) :: List.map (fun w -> y :: w) zs)
      in
      let rec flat_map f = function
        | [] -> []
        | x :: xs -> f x @ flat_map f xs
      in
      match lst with
      | [] -> [[]]
      | x :: xs -> 
          flat_map (insert_all x) (permutations xs)

permutations [1; 2; 3]
|}

let a80 = "The answer is [[1;2;3];[2;1;3];[2;3;1];[1;3;2];[3;1;2];[3;2;1]]. Process:
1. Gets permutations of [2;3]
2. Inserts 1 at every possible position
3. Combines all results
Shows generating all permutations recursively"

(* Medium-level Questions using List Module *)

let q81 = {|
What does this evaluate to?

let nums = [1; 2; 3; 4; 5]
let result = List.fold_left (fun acc x -> 
  if x mod 2 = 0 
  then x :: acc 
  else acc
) [] nums |> List.rev

result
|}

let a81 = "The answer is [2; 4]. Here's why:
1. fold_left accumulates even numbers in reverse: [4; 2]
2. List.rev gives final result: [2; 4]
Shows combining fold_left with rev for filtering"

let q82 = {|
What's the result?

let words = ["hello"; "world"; "ocaml"; "fun"]
let result = List.filter (fun s -> 
  List.exists (fun c -> c = 'o') (List.init (String.length s) (String.get s))
) words

result
|}

let a82 = "The answer is [\"hello\"; \"world\"; \"ocaml\"]. Analysis:
1. List.init creates list of characters for each string
2. List.exists checks if 'o' is in that list
3. List.filter keeps strings containing 'o'
Shows composition of List functions"

let q83 = {|
What does this evaluate to?

let nums = [1; 2; 3; 4; 5]
let result = List.mapi (fun i x -> 
  if i mod 2 = 0 then x else -x
) nums

result
|}

let a83 = "The answer is [1; -2; 3; -4; 5]. Here's why:
1. Index 0: 1 unchanged
2. Index 1: -2 becomes negative
3. Index 2: 3 remains positive
4. Index 3: -4 becomes negative
5. Index 4: 5 remains positive
Shows using mapi to access indices"

let q84 = {|
What's the result?

let lists = [[1; 2]; [3; 4]; [5; 6]]
let result = List.fold_left (fun acc lst ->
  acc @ [List.hd lst + List.hd (List.rev lst)]
) [] lists

result
|}

let a84 = "The answer is [3; 7; 11]. Calculation:
1. [1; 2]: 1 + 2 = 3
2. [3; 4]: 3 + 4 = 7
3. [5; 6]: 5 + 6 = 11
Shows combining fold_left with list operations"

let q85 = {|
What does this evaluate to?

let nums = [1; 2; 3; 4; 5]
let result = List.fold_right (fun x acc ->
  match acc with
  | [] -> [x]
  | hd::_ when x > hd -> x :: acc
  | _ -> acc
) nums []

result
|}

let a85 = "The answer is [5; 4]. Here's why:
1. Processes right to left
2. First adds 5
3. Then adds 4 (> previous element)
4. 3, 2, 1 are all less than 4, so skipped
Shows fold_right with conditional accumulation"

let q86 = {|
What's the result?

let pairs = [(1, "one"); (2, "two"); (3, "three")]
let result = 
  List.filter (fun (n, _) -> n mod 2 = 1) pairs
  |> List.map snd
  |> List.fold_left (^) ""

result
|}

let a86 = "The answer is \"onethree\". Steps:
1. filter keeps (1,\"one\") and (3,\"three\")
2. map extracts [\"one\"; \"three\"]
3. fold_left concatenates strings
Shows chaining List operations"

let q87 = {|
What does this evaluate to?

let matrix = [[1; 2; 3]; [4; 5; 6]]
let result = List.map (fun row ->
  List.mapi (fun i x -> x * (i + 1)) row
) matrix

result
|}

let a87 = "The answer is [[1; 4; 9]; [4; 10; 18]]. Calculation:
1. First row: [1*1; 2*2; 3*3]
2. Second row: [4*1; 5*2; 6*3]
Shows nested List operations on 2D list"

let q88 = {|
What's the result?

let nums = [2; -1; 3; -4; 5; -6]
let result = List.fold_left (fun (pos, neg) x ->
  if x > 0 
  then (pos + x, neg)
  else (pos, neg + x)
) (0, 0) nums

result
|}

let a88 = "The answer is (10, -11). Accumulation:
1. Positive sum: 2 + 3 + 5 = 10
2. Negative sum: (-1) + (-4) + (-6) = -11
Shows fold_left with tuple accumulator"

let q89 = {|
What does this evaluate to?

let words = ["cat"; "dog"; "bird"]
let result = Array.map (fun s ->
  let a = Array.make (String.length s) ' ' in
  String.iteri (fun i c -> a.(i) <- c) s;
  Array.fold_right (fun c acc -> 
    if c = 'o' then acc + 1 else acc
  ) a 0
) words

Array.to_list result
|}

let a89 = "The answer is [1; 1; 1]. Explanation:
1. Converts each string to char array
2. Counts 'o' occurrences in each string
3. \"hello\", \"world\", \"ocaml\" each have one 'o'
Shows combining string and array operations"

let q90 = {|
What's the result?

let nums = [1; 2; 3; 4; 5]
let result = List.fold_left (fun acc x ->
  match acc with
  | [] -> [[x]]
  | hd::tl -> 
      if List.length hd < 2 
      then (x::hd)::tl
      else [x]::acc
) [] (List.rev nums)

result
|}

let a90 = "The answer is [[5; 4]; [3; 2]; [1]]. Here's why:
1. Processes [5;4;3;2;1]
2. Groups into pairs from left
3. Last element forms singleton
Shows complex list grouping with fold_left"

(* Medium-level Questions using Array Module *)

let q91 = {|
What does this evaluate to?

let arr = Array.init 5 (fun i -> i * 2)
let result = Array.fold_left (fun acc x -> 
  if x mod 3 = 0 then acc + x else acc
) 0 arr

result
|}

let a91 = "The answer is 6. Here's why:
1. Array contains [|0; 2; 4; 6; 8|]
2. Only 0 and 6 are divisible by 3
3. 0 + 6 = 6
Shows Array.init and fold_left with condition"

let q92 = {|
What's the result?

let matrix = Array.make_matrix 3 3 0
let _ = 
  for i = 0 to 2 do
    for j = 0 to 2 do
      matrix.(i).(j) <- if i = j then 1 else 0
    done
  done

Array.map Array.to_list matrix |> Array.to_list
|}

let a92 = "The answer is [[1; 0; 0]; [0; 1; 0]; [0; 0; 1]]. Process:
1. Creates 3x3 matrix of zeros
2. Sets diagonal elements to 1
3. Converts to list of lists
Shows 2D array manipulation"

let q93 = {|
What does this evaluate to?

let arr = [|1; 3; 5; 7; 9|]
let result = Array.mapi (fun i x ->
  if i mod 2 = 0 then x else x * 2
) arr

Array.to_list result
|}

let a93 = "The answer is [1; 6; 5; 14; 9]. Here's why:
1. Index 0: 1 unchanged
2. Index 1: 3 * 2 = 6
3. Index 2: 5 unchanged
4. Index 3: 7 * 2 = 14
5. Index 4: 9 unchanged
Shows Array.mapi with index-based transformation"

let q94 = {|
What's the result?

let arr = Array.init 6 (fun i -> i + 1)
let _ = Array.sort compare arr
let result = Array.sub arr 2 3

Array.to_list result
|}

let a94 = "The answer is [3; 4; 5]. Analysis:
1. Initial array: [|1; 2; 3; 4; 5; 6|]
2. Array.sort has no effect (already sorted)
3. Array.sub extracts 3 elements starting at index 2
Shows Array.sub operation"

let q95 = {|
What does this evaluate to?

let arr1 = [|1; 2; 3|]
let arr2 = [|4; 5; 6|]
let result = Array.append arr1 arr2
let _ = Array.blit arr1 1 result 4 2

Array.to_list result
|}

let a95 = "The answer is [1; 2; 3; 4; 2; 3]. Steps:
1. Array.append creates [|1; 2; 3; 4; 5; 6|]
2. Array.blit copies 2 elements from arr1[1] to result[4]
Shows Array.append and Array.blit operations"

let q96 = {|
What's the result?

let arr = [|"hello"; "world"; "ocaml"|]
let result = Array.map (fun s ->
  let a = Array.make (String.length s) ' ' in
  String.iteri (fun i c -> a.(i) <- c) s;
  Array.fold_right (fun c acc -> 
    if c = 'o' then acc + 1 else acc
  ) a 0
) arr

Array.to_list result
|}

let a96 = "The answer is [1; 1; 1]. Explanation:
1. Converts each string to char array
2. Counts 'o' occurrences in each string
3. \"hello\", \"world\", \"ocaml\" each have one 'o'
Shows combining string and array operations"

let q97 = {|
What does this evaluate to?

let arr = Array.make 5 []
let _ = 
  for i = 0 to 4 do
    arr.(i) <- List.init i (fun x -> x * 2)
  done

Array.to_list arr
|}

let a97 = "The answer is [[];[0];[0;2];[0;2;4];[0;2;4;6]]. Process:
1. Creates array of empty lists
2. Each index i gets list of first i even numbers
3. Shows combining Array and List operations
Shows array mutation with list generation"

let q98 = {|
What's the result?

let matrix = Array.make_matrix 3 3 0
let _ = 
  for i = 0 to 2 do
    matrix.(i) <- Array.init 3 (fun j -> i + j)
  done

let result = Array.map (Array.fold_left (+) 0) matrix
Array.to_list result
|}

let a98 = "The answer is [3; 6; 9]. Here's why:
1. Matrix rows: [|0;1;2|], [|1;2;3|], [|2;3;4|]
2. Sum of first row: 0+1+2 = 3
3. Sum of second row: 1+2+3 = 6
4. Sum of third row: 2+3+4 = 9
Shows 2D array manipulation with fold"

let q99 = {|
What does this evaluate to?

let arr = [|1; 2; 3; 4; 5|]
let result = Array.make (Array.length arr) 0
let _ = 
  for i = 0 to Array.length arr - 1 do
    result.(i) <- Array.fold_left (+) 0 (
      Array.sub arr 0 (i + 1)
    )
  done

Array.to_list result
|}

let a99 = "The answer is [1; 3; 6; 10; 15]. Calculation:
1. Index 0: sum of arr[0..0] = 1
2. Index 1: sum of arr[0..1] = 1+2 = 3
3. Index 2: sum of arr[0..2] = 1+2+3 = 6
4. Index 3: sum of arr[0..3] = 1+2+3+4 = 10
5. Index 4: sum of arr[0..4] = 1+2+3+4+5 = 15
Shows running sum using array operations"

let q100 = {|
What's the result?

let arr = [|"a"; "b"; "c"; "d"|]
let result = Array.make (Array.length arr) ""
let _ = 
  for i = Array.length arr - 1 downto 0 do
    result.(Array.length arr - 1 - i) <- 
      Array.fold_right (^) (Array.sub arr 0 (i + 1)) ""
  done

Array.to_list result
|}

let a100 = "The answer is [\"a\"; \"ab\"; \"abc\"; \"abcd\"]. Process:
1. Creates prefixes of original array
2. Concatenates strings in each prefix
3. Stores results in reverse order
Shows string concatenation with array operations"

(* Questions about Type Inference *)

let q101 = {|
What is the type of this function?

let f x y = 
  if x > y then x else y
|}

let a101 = "Type: 'a -> 'a -> 'a where 'a has comparison
Explanation:
1. > operator requires comparable types
2. Both branches return same type as inputs
3. Since only comparison is used, type is polymorphic
4. Common examples: int -> int -> int or float -> float -> float"

let q102 = {|
What is the type of this function?

let rec f = function
  | [] -> []
  | x::xs -> (x, String.length x) :: f xs
|}

let a102 = "Type: string list -> (string * int) list
Explanation:
1. Input must be string list due to String.length
2. Each element is paired with its length
3. Output is list of string-int pairs
Shows list processing with type constraints"

let q103 = {|
What is the type of this function?

let f g x y = g (x, y)
|}

let a103 = "Type: ('a * 'b -> 'c) -> 'a -> 'b -> 'c
Explanation:
1. g takes a tuple argument
2. x and y are paired into tuple
3. Result type depends on g's return type
4. All types are polymorphic
Shows higher-order function with tupling"

let q104 = {|
What is the type of this function?

let rec f x = function
  | [] -> x
  | h::t -> h (f x t)
|}

let a104 = "Type: 'a -> ('a -> 'a) list -> 'a
Explanation:
1. x has type 'a
2. Each h must be function 'a -> 'a
3. Result has same type as x
4. Shows recursive function composition"

let q105 = {|
What is the type of this function?

let f = function
  | None -> []
  | Some x -> [x; x]
|}

let a105 = "Type: 'a option -> 'a list
Explanation:
1. Input is option type
2. Output is list containing two copies if Some
3. Empty list if None
4. Type is polymorphic as no operations on x"

let q106 = {|
What is the type of this function?

let f x y = 
  let g a = a y in
  let h b = b x in
  g (h (fun a b -> a))
|}

let a106 = "Type: 'a -> 'b -> 'a
Explanation:
1. Complex function composition
2. Inner function returns first argument
3. Outer applications preserve type
4. Result type matches first input
Shows nested function application"

let q107 = {|
What is the type of this function?

let rec f g = function
  | [] -> g
  | x::xs -> fun y -> f (g y) xs
|}

let a107 = "Type: ('a -> 'b) -> 'a list -> 'a -> 'b
Explanation:
1. g is function 'a -> 'b
2. Input list length determines number of applications
3. Each element triggers new function application
Shows complex higher-order function"

let q108 = {|
What is the type of this function?

let f x = 
  let (a, b) = x in
  if a then [b] else []
|}

let a108 = "Type: bool * 'a -> 'a list
Explanation:
1. First component must be bool (used in if)
2. Second component can be any type
3. Result is list of second component's type
Shows pattern matching with tuples"

let q109 = {|
What is the type of this function?

let rec f = function
  | x::xs when x > 0 -> x :: f xs
  | _::xs -> f xs
  | [] -> []
|}

let a109 = "Type: int list -> int list
Explanation:
1. Comparison with 0 requires int type
2. Returns list of same type as input
3. Guard uses integer comparison
Shows list filtering with type constraint"

let q110 = {|
What is the type of this function?

let f x y =
  let g a = a y in
  let h b = b x in
  g (h (fun a b -> a))
|}

let a110 = "Type: 'a -> 'b -> 'a
Explanation:
1. Complex function composition
2. Inner function returns first argument
3. Outer applications preserve type
4. Result type matches first input
Shows nested function application"

(* Helper function to print questions and answers *)
let print_qa n q a =
  Printf.printf "\nQuestion %d:\n%s\n\nAnswer:\n%s\n" n q a

let () =
  print_qa 1 q1 a1;
  print_qa 2 q2 a2;
  print_qa 3 q3 a3;
  print_qa 4 q4 a4;
  print_qa 5 q5 a5;
  print_qa 6 q6 a6;
  print_qa 7 q7 a7;
  print_qa 8 q8 a8;
  print_qa 9 q9 a9;
  print_qa 10 q10 a10;
  print_qa 11 q11 a11;
  print_qa 12 q12 a12;
  print_qa 13 q13 a13;
  print_qa 14 q14 a14;
  print_qa 15 q15 a15;
  print_qa 16 q16 a16;
  print_qa 17 q17 a17;
  print_qa 18 q18 a18;
  print_qa 19 q19 a19;
  print_qa 20 q20 a20;
  print_qa 21 q21 a21;
  print_qa 22 q22 a22;
  print_qa 23 q23 a23;
  print_qa 24 q24 a24;
  print_qa 25 q25 a25;
  print_qa 26 q26 a26;
  print_qa 27 q27 a27;
  print_qa 28 q28 a28;
  print_qa 29 q29 a29;
  print_qa 30 q30 a30;
  print_qa 31 q31 a31;
  print_qa 32 q32 a32;
  print_qa 33 q33 a33;
  print_qa 34 q34 a34;
  print_qa 35 q35 a35;
  print_qa 36 q36 a36;
  print_qa 37 q37 a37;
  print_qa 38 q38 a38;
  print_qa 39 q39 a39;
  print_qa 40 q40 a40;
  print_qa 61 q61 a61;
  print_qa 62 q62 a62;
  print_qa 63 q63 a63;
  print_qa 64 q64 a64;
  print_qa 65 q65 a65;
  print_qa 66 q66 a66;
  print_qa 67 q67 a67;
  print_qa 68 q68 a68;
  print_qa 69 q69 a69;
  print_qa 70 q70 a70;
  print_qa 71 q71 a71;
  print_qa 72 q72 a72;
  print_qa 73 q73 a73;
  print_qa 74 q74 a74;
  print_qa 75 q75 a75;
  print_qa 76 q76 a76;
  print_qa 77 q77 a77;
  print_qa 78 q78 a78;
  print_qa 79 q79 a79;
  print_qa 80 q80 a80;
  print_qa 91 q91 a91;
  print_qa 92 q92 a92;
  print_qa 93 q93 a93;
  print_qa 94 q94 a94;
  print_qa 95 q95 a95;
  print_qa 96 q96 a96;
  print_qa 97 q97 a97;
  print_qa 98 q98 a98;
  print_qa 99 q99 a99;
  print_qa 100 q100 a100;
  print_qa 101 q101 a101;
  print_qa 102 q102 a102;
  print_qa 103 q103 a103;
  print_qa 104 q104 a104;
  print_qa 105 q105 a105;
  print_qa 106 q106 a106;
  print_qa 107 q107 a107;
  print_qa 108 q108 a108;
  print_qa 109 q109 a109;
  print_qa 110 q110 a110
