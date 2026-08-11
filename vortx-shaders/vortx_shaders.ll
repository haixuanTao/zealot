; ModuleID = 'builtin.module'
source_filename = "vortx_shaders"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_16 = addrspace(3) global [1040 x float] undef, align 4
@__shared_mem_15 = addrspace(3) global [1088 x float] undef, align 4
@__shared_mem_14 = addrspace(3) global [1040 x float] undef, align 4
@__shared_mem_13 = addrspace(3) global [1088 x float] undef, align 4
@__shared_mem_12 = addrspace(3) global [128 x i32] undef, align 4
@__shared_mem_11 = addrspace(3) global [128 x float] undef, align 4
@__shared_mem_10 = addrspace(3) global [128 x i32] undef, align 4
@__shared_mem_9 = addrspace(3) global [128 x i32] undef, align 4
@__shared_mem_8 = addrspace(3) global [128 x float] undef, align 4
@__shared_mem_7 = addrspace(3) global [128 x float] undef, align 4
@__shared_mem_6 = addrspace(3) global [128 x i32] undef, align 4
@__shared_mem_5 = addrspace(3) global [128 x float] undef, align 4
@__shared_mem_4 = addrspace(3) global [128 x float] undef, align 4
@__shared_mem_3 = addrspace(3) global [128 x i32] undef, align 4
@__shared_mem_2 = addrspace(3) global [128 x i32] undef, align 4
@__shared_mem_1 = addrspace(3) global [128 x i32] undef, align 4
@__shared_mem_0 = addrspace(3) global [128 x i32] undef, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()

define void @reduce_max_i32_cuda_entry_a23ce822c89c7cd7(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb12
bb1:
  %v15 = phi i32 [ %v80, %bb16 ], [ %v97, %bb26 ]
  %v16 = phi i32 [ %v83, %bb16 ], [ %v98, %bb26 ]
  %v17 = add i64 %v85, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb22, label %bb21
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v102, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to i32*
  %v150 = getelementptr inbounds i32, i32* %v149, i64 %v37
  %v41 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v41 to i32*
  %v42 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to i32 addrspace(3)*
  %v156 = getelementptr inbounds i32, i32 addrspace(3)* %v155, i64 %v66
  %v46 = bitcast i32 addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb25
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb24
bb5:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v48 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v48 to i8 addrspace(3)**
  %v49 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v160 = bitcast i8 addrspace(3)* %v49 to i32 addrspace(3)*
  %v161 = getelementptr inbounds i32, i32 addrspace(3)* %v160, i64 %v66
  %v50 = bitcast i32 addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb26
bb6:
  %v51 = phi i64 [ %v57, %bb9 ], [ 64, %bb24 ]
  %v52 = phi i32 [ %v121, %bb9 ], [ 0, %bb24 ]
  %v53 = icmp ult i32 %v52, 7
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb28, label %bb27
bb7:
  call void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_max_i32_cuda_entry_a23ce822c89c7cd720reduce_workspace_maxINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB8_(i64 %v66, i64 %v51, i8* %v12) #0
  br label %bb9
bb8:
  %v56 = icmp eq i32 %v14, 0
  br i1 %v56, label %bb10, label %bb11
bb9:
  %v57 = udiv i64 %v51, 2
  br label %bb6
bb10:
  %v162 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v163 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v162, i32 0, i32 0
  %v58 = bitcast i8 addrspace(3)** %v163 to i8*
  %v164 = bitcast i8* %v58 to i8 addrspace(3)**
  %v59 = load i8 addrspace(3)*, i8 addrspace(3)** %v164, align 8
  %v60 = getelementptr i8, i8 addrspace(3)* %v59, i64 0
  %v165 = bitcast i8 addrspace(3)* %v60 to i32 addrspace(3)*
  %v166 = getelementptr inbounds i32, i32 addrspace(3)* %v165, i64 0
  %v61 = bitcast i32 addrspace(3)* %v166 to i8 addrspace(3)*
  br label %bb31
bb11:
  ret void
bb12:
  %v62 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb13
bb13:
  %v63 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb14
bb14:
  %v64 = bitcast [128 x i32] addrspace(3)* @__shared_mem_0 to i8 addrspace(3)*
  %v65 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v64, 0
  %v168 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v65, { i8 addrspace(3)* }* %v168, align 8
  %v66 = zext i32 %v14 to i64
  %v169 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v170 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v169, i32 0, i32 0
  %v67 = bitcast i8 addrspace(3)** %v170 to i8*
  %v171 = bitcast i8* %v67 to i8 addrspace(3)**
  %v68 = load i8 addrspace(3)*, i8 addrspace(3)** %v171, align 8
  %v172 = bitcast i8 addrspace(3)* %v68 to i32 addrspace(3)*
  %v173 = getelementptr inbounds i32, i32 addrspace(3)* %v172, i64 %v66
  %v69 = bitcast i32 addrspace(3)* %v173 to i8 addrspace(3)*
  br label %bb15
bb15:
  %v174 = bitcast i8 addrspace(3)* %v69 to i32 addrspace(3)*
  store i32 2147483648, i32 addrspace(3)* %v174, align 4
  %v70 = trunc i64 %v66 to i32
  %v175 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v176 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v175, i32 0, i32 3
  %v71 = bitcast i32* %v176 to i8*
  %v177 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v177, align 4
  %v73 = insertvalue { i32, i32 } undef, i32 %v70, 0
  %v74 = insertvalue { i32, i32 } %v73, i32 %v72, 1
  %v75 = extractvalue { i32, i32 } %v74, 0
  %v76 = extractvalue { i32, i32 } %v74, 1
  %v77 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v75, i32 %v76, i64 128) #0
  %v179 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v77, { { i32, i32 }, i64, i1, [7 x i8] }* %v179, align 8
  br label %bb16
bb16:
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 0
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 0
  %v81 = bitcast { i32, i32 }* %v186 to i8*
  %v187 = bitcast i8* %v81 to { i32, i32 }*
  %v188 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v187, i32 0, i32 1
  %v82 = bitcast i32* %v188 to i8*
  %v189 = bitcast i8* %v82 to i32*
  %v83 = load i32, i32* %v189, align 4
  %v190 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v191 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v190, i32 0, i32 1
  %v84 = bitcast i64* %v191 to i8*
  %v192 = bitcast i8* %v84 to i64*
  %v85 = load i64, i64* %v192, align 8
  %v193 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v194 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v193, i32 0, i32 2
  %v86 = bitcast i1* %v194 to i8*
  %v195 = bitcast i8* %v86 to i1*
  %v87 = load i1, i1* %v195, align 1
  br label %bb1
bb17:
  %v88 = add i32 %v15, %v108
  %v89 = sub i32 %v16, 1
  %v90 = insertvalue { i32, i32 } undef, i32 1, 0
  %v91 = insertvalue { i32, i32 } %v90, i32 %v15, 1
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb19
bb18:
  %v94 = insertvalue { i32, i32 } undef, i32 0, 0
  %v95 = extractvalue { i32, i32 } %v94, 0
  %v96 = extractvalue { i32, i32 } %v94, 1
  br label %bb19
bb19:
  %v97 = phi i32 [ %v88, %bb17 ], [ %v15, %bb18 ]
  %v98 = phi i32 [ %v89, %bb17 ], [ %v16, %bb18 ]
  %v99 = phi i32 [ %v92, %bb17 ], [ %v95, %bb18 ]
  %v100 = phi i32 [ %v93, %bb17 ], [ %v96, %bb18 ]
  %v101 = insertvalue { i32, i32 } undef, i32 %v99, 0
  %v102 = insertvalue { i32, i32 } %v101, i32 %v100, 1
  %v103 = extractvalue { i32, i32 } %v102, 0
  %v104 = zext i32 %v103 to i64
  %v105 = icmp eq i64 %v104, 0
  br i1 %v105, label %bb4, label %bb20
bb20:
  %v106 = icmp eq i64 %v104, 1
  br i1 %v106, label %bb3, label %bb2
bb21:
  br label %bb23
bb22:
  %v107 = trunc i64 %v30 to i32
  br label %bb23
bb23:
  %v108 = phi i32 [ 4294967295, %bb21 ], [ %v107, %bb22 ]
  %v109 = icmp ugt i32 %v16, 0
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb18, label %bb17
bb24:
  br label %bb6
bb25:
  %v199 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v111 = load i32, i32 addrspace(3)* %v199, align 4
  %v112 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCslfDnHtpJyg4_13vortx_shaders(i32 %v111, i32 %v42) #0
  br label %bb5
bb26:
  %v200 = bitcast i8 addrspace(3)* %v50 to i32 addrspace(3)*
  store i32 %v112, i32 addrspace(3)* %v200, align 4
  br label %bb1
bb27:
  %v113 = add i32 %v52, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v52, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb29
bb28:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb29
bb29:
  %v121 = phi i32 [ %v113, %bb27 ], [ %v52, %bb28 ]
  %v122 = phi i32 [ %v116, %bb27 ], [ %v119, %bb28 ]
  %v123 = phi i32 [ %v117, %bb27 ], [ %v120, %bb28 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb8, label %bb30
bb30:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb7, label %bb2
bb31:
  %v204 = bitcast i8 addrspace(3)* %v61 to i32 addrspace(3)*
  %v130 = load i32, i32 addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to i32*
  %v206 = getelementptr inbounds i32, i32* %v205, i64 0
  %v132 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v132 to i32*
  store i32 %v130, i32* %v207, align 4
  br label %bb11
}

define void @reduce_add_u32_cuda_entry_22ce7ff7f5eb01d4(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb25 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb25 ]
  %v17 = add i64 %v82, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to i32*
  %v150 = getelementptr inbounds i32, i32* %v149, i64 %v37
  %v41 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v41 to i32*
  %v42 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to i32 addrspace(3)*
  %v156 = getelementptr inbounds i32, i32 addrspace(3)* %v155, i64 %v63
  %v46 = bitcast i32 addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v121, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb27, label %bb26
bb6:
  call void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_summINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB6_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v160 = bitcast i8 addrspace(3)* %v57 to i32 addrspace(3)*
  %v161 = getelementptr inbounds i32, i32 addrspace(3)* %v160, i64 0
  %v58 = bitcast i32 addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb30
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x i32] addrspace(3)* @__shared_mem_1 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v163, align 8
  %v63 = zext i32 %v14 to i64
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v165 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v164, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v165 to i8*
  %v166 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v166, align 8
  %v167 = bitcast i8 addrspace(3)* %v65 to i32 addrspace(3)*
  %v168 = getelementptr inbounds i32, i32 addrspace(3)* %v167, i64 %v63
  %v66 = bitcast i32 addrspace(3)* %v168 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v169 = bitcast i8 addrspace(3)* %v66 to i32 addrspace(3)*
  store i32 0, i32 addrspace(3)* %v169, align 4
  %v67 = trunc i64 %v63 to i32
  %v170 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v171 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v170, i32 0, i32 3
  %v68 = bitcast i32* %v171 to i8*
  %v172 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v172, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v174 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, align 8
  br label %bb15
bb15:
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v176 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v176 to i8*
  %v177 = bitcast i8* %v75 to { i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v177, i32 0, i32 0
  %v76 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 1
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 1
  %v81 = bitcast i64* %v186 to i8*
  %v187 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v187, align 8
  %v188 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v189 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v188, i32 0, i32 2
  %v83 = bitcast i1* %v189 to i8*
  %v190 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v190, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v194 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v108 = load i32, i32 addrspace(3)* %v194, align 4
  %v109 = add i32 %v108, %v42
  %v195 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v196 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v195, i32 0, i32 0
  %v110 = bitcast i8 addrspace(3)** %v196 to i8*
  %v197 = bitcast i8* %v110 to i8 addrspace(3)**
  %v111 = load i8 addrspace(3)*, i8 addrspace(3)** %v197, align 8
  %v198 = bitcast i8 addrspace(3)* %v111 to i32 addrspace(3)*
  %v199 = getelementptr inbounds i32, i32 addrspace(3)* %v198, i64 %v63
  %v112 = bitcast i32 addrspace(3)* %v199 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v200 = bitcast i8 addrspace(3)* %v112 to i32 addrspace(3)*
  store i32 %v109, i32 addrspace(3)* %v200, align 4
  br label %bb1
bb26:
  %v113 = add i32 %v49, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v49, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb28
bb27:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb28
bb28:
  %v121 = phi i32 [ %v113, %bb26 ], [ %v49, %bb27 ]
  %v122 = phi i32 [ %v116, %bb26 ], [ %v119, %bb27 ]
  %v123 = phi i32 [ %v117, %bb26 ], [ %v120, %bb27 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb7, label %bb29
bb29:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb6, label %bb2
bb30:
  %v204 = bitcast i8 addrspace(3)* %v58 to i32 addrspace(3)*
  %v130 = load i32, i32 addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to i32*
  %v206 = getelementptr inbounds i32, i32* %v205, i64 0
  %v132 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v132 to i32*
  store i32 %v130, i32* %v207, align 4
  br label %bb10
}

define void @reduce_mul_u32_cuda_entry_453e89c2a77c7e7a(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb25 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb25 ]
  %v17 = add i64 %v82, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to i32*
  %v150 = getelementptr inbounds i32, i32* %v149, i64 %v37
  %v41 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v41 to i32*
  %v42 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to i32 addrspace(3)*
  %v156 = getelementptr inbounds i32, i32 addrspace(3)* %v155, i64 %v63
  %v46 = bitcast i32 addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v121, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb27, label %bb26
bb6:
  call void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_mulmINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB6_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v160 = bitcast i8 addrspace(3)* %v57 to i32 addrspace(3)*
  %v161 = getelementptr inbounds i32, i32 addrspace(3)* %v160, i64 0
  %v58 = bitcast i32 addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb30
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x i32] addrspace(3)* @__shared_mem_2 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v163, align 8
  %v63 = zext i32 %v14 to i64
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v165 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v164, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v165 to i8*
  %v166 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v166, align 8
  %v167 = bitcast i8 addrspace(3)* %v65 to i32 addrspace(3)*
  %v168 = getelementptr inbounds i32, i32 addrspace(3)* %v167, i64 %v63
  %v66 = bitcast i32 addrspace(3)* %v168 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v169 = bitcast i8 addrspace(3)* %v66 to i32 addrspace(3)*
  store i32 1, i32 addrspace(3)* %v169, align 4
  %v67 = trunc i64 %v63 to i32
  %v170 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v171 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v170, i32 0, i32 3
  %v68 = bitcast i32* %v171 to i8*
  %v172 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v172, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v174 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, align 8
  br label %bb15
bb15:
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v176 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v176 to i8*
  %v177 = bitcast i8* %v75 to { i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v177, i32 0, i32 0
  %v76 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 1
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 1
  %v81 = bitcast i64* %v186 to i8*
  %v187 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v187, align 8
  %v188 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v189 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v188, i32 0, i32 2
  %v83 = bitcast i1* %v189 to i8*
  %v190 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v190, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v194 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v108 = load i32, i32 addrspace(3)* %v194, align 4
  %v109 = mul i32 %v108, %v42
  %v195 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v196 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v195, i32 0, i32 0
  %v110 = bitcast i8 addrspace(3)** %v196 to i8*
  %v197 = bitcast i8* %v110 to i8 addrspace(3)**
  %v111 = load i8 addrspace(3)*, i8 addrspace(3)** %v197, align 8
  %v198 = bitcast i8 addrspace(3)* %v111 to i32 addrspace(3)*
  %v199 = getelementptr inbounds i32, i32 addrspace(3)* %v198, i64 %v63
  %v112 = bitcast i32 addrspace(3)* %v199 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v200 = bitcast i8 addrspace(3)* %v112 to i32 addrspace(3)*
  store i32 %v109, i32 addrspace(3)* %v200, align 4
  br label %bb1
bb26:
  %v113 = add i32 %v49, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v49, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb28
bb27:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb28
bb28:
  %v121 = phi i32 [ %v113, %bb26 ], [ %v49, %bb27 ]
  %v122 = phi i32 [ %v116, %bb26 ], [ %v119, %bb27 ]
  %v123 = phi i32 [ %v117, %bb26 ], [ %v120, %bb27 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb7, label %bb29
bb29:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb6, label %bb2
bb30:
  %v204 = bitcast i8 addrspace(3)* %v58 to i32 addrspace(3)*
  %v130 = load i32, i32 addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to i32*
  %v206 = getelementptr inbounds i32, i32* %v205, i64 0
  %v132 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v132 to i32*
  store i32 %v130, i32* %v207, align 4
  br label %bb10
}

define void @reduce_mul_i32_cuda_entry_aa0d5c1e4a461d98(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb25 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb25 ]
  %v17 = add i64 %v82, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to i32*
  %v150 = getelementptr inbounds i32, i32* %v149, i64 %v37
  %v41 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v41 to i32*
  %v42 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to i32 addrspace(3)*
  %v156 = getelementptr inbounds i32, i32 addrspace(3)* %v155, i64 %v63
  %v46 = bitcast i32 addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v121, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb27, label %bb26
bb6:
  call void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_mullINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB6_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v160 = bitcast i8 addrspace(3)* %v57 to i32 addrspace(3)*
  %v161 = getelementptr inbounds i32, i32 addrspace(3)* %v160, i64 0
  %v58 = bitcast i32 addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb30
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x i32] addrspace(3)* @__shared_mem_3 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v163, align 8
  %v63 = zext i32 %v14 to i64
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v165 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v164, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v165 to i8*
  %v166 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v166, align 8
  %v167 = bitcast i8 addrspace(3)* %v65 to i32 addrspace(3)*
  %v168 = getelementptr inbounds i32, i32 addrspace(3)* %v167, i64 %v63
  %v66 = bitcast i32 addrspace(3)* %v168 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v169 = bitcast i8 addrspace(3)* %v66 to i32 addrspace(3)*
  store i32 1, i32 addrspace(3)* %v169, align 4
  %v67 = trunc i64 %v63 to i32
  %v170 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v171 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v170, i32 0, i32 3
  %v68 = bitcast i32* %v171 to i8*
  %v172 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v172, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v174 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, align 8
  br label %bb15
bb15:
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v176 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v176 to i8*
  %v177 = bitcast i8* %v75 to { i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v177, i32 0, i32 0
  %v76 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 1
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 1
  %v81 = bitcast i64* %v186 to i8*
  %v187 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v187, align 8
  %v188 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v189 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v188, i32 0, i32 2
  %v83 = bitcast i1* %v189 to i8*
  %v190 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v190, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v194 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v108 = load i32, i32 addrspace(3)* %v194, align 4
  %v109 = mul i32 %v108, %v42
  %v195 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v196 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v195, i32 0, i32 0
  %v110 = bitcast i8 addrspace(3)** %v196 to i8*
  %v197 = bitcast i8* %v110 to i8 addrspace(3)**
  %v111 = load i8 addrspace(3)*, i8 addrspace(3)** %v197, align 8
  %v198 = bitcast i8 addrspace(3)* %v111 to i32 addrspace(3)*
  %v199 = getelementptr inbounds i32, i32 addrspace(3)* %v198, i64 %v63
  %v112 = bitcast i32 addrspace(3)* %v199 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v200 = bitcast i8 addrspace(3)* %v112 to i32 addrspace(3)*
  store i32 %v109, i32 addrspace(3)* %v200, align 4
  br label %bb1
bb26:
  %v113 = add i32 %v49, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v49, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb28
bb27:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb28
bb28:
  %v121 = phi i32 [ %v113, %bb26 ], [ %v49, %bb27 ]
  %v122 = phi i32 [ %v116, %bb26 ], [ %v119, %bb27 ]
  %v123 = phi i32 [ %v117, %bb26 ], [ %v120, %bb27 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb7, label %bb29
bb29:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb6, label %bb2
bb30:
  %v204 = bitcast i8 addrspace(3)* %v58 to i32 addrspace(3)*
  %v130 = load i32, i32 addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to i32*
  %v206 = getelementptr inbounds i32, i32* %v205, i64 0
  %v132 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v132 to i32*
  store i32 %v130, i32* %v207, align 4
  br label %bb10
}

define void @reduce_add_f32_cuda_entry_bd874dedbb24ff35(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb25 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb25 ]
  %v17 = add i64 %v82, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to float*
  %v150 = getelementptr inbounds float, float* %v149, i64 %v37
  %v41 = bitcast float* %v150 to i8*
  %v151 = bitcast i8* %v41 to float*
  %v42 = load float, float* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to float addrspace(3)*
  %v156 = getelementptr inbounds float, float addrspace(3)* %v155, i64 %v63
  %v46 = bitcast float addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v121, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb27, label %bb26
bb6:
  call void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_sumfINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB6_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v160 = bitcast i8 addrspace(3)* %v57 to float addrspace(3)*
  %v161 = getelementptr inbounds float, float addrspace(3)* %v160, i64 0
  %v58 = bitcast float addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb30
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x float] addrspace(3)* @__shared_mem_4 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v163, align 8
  %v63 = zext i32 %v14 to i64
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v165 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v164, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v165 to i8*
  %v166 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v166, align 8
  %v167 = bitcast i8 addrspace(3)* %v65 to float addrspace(3)*
  %v168 = getelementptr inbounds float, float addrspace(3)* %v167, i64 %v63
  %v66 = bitcast float addrspace(3)* %v168 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v169 = bitcast i8 addrspace(3)* %v66 to float addrspace(3)*
  store float 0.0, float addrspace(3)* %v169, align 4
  %v67 = trunc i64 %v63 to i32
  %v170 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v171 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v170, i32 0, i32 3
  %v68 = bitcast i32* %v171 to i8*
  %v172 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v172, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v174 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, align 8
  br label %bb15
bb15:
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v176 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v176 to i8*
  %v177 = bitcast i8* %v75 to { i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v177, i32 0, i32 0
  %v76 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 1
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 1
  %v81 = bitcast i64* %v186 to i8*
  %v187 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v187, align 8
  %v188 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v189 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v188, i32 0, i32 2
  %v83 = bitcast i1* %v189 to i8*
  %v190 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v190, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v194 = bitcast i8 addrspace(3)* %v46 to float addrspace(3)*
  %v108 = load float, float addrspace(3)* %v194, align 4
  %v109 = fadd contract float %v108, %v42
  %v195 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v196 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v195, i32 0, i32 0
  %v110 = bitcast i8 addrspace(3)** %v196 to i8*
  %v197 = bitcast i8* %v110 to i8 addrspace(3)**
  %v111 = load i8 addrspace(3)*, i8 addrspace(3)** %v197, align 8
  %v198 = bitcast i8 addrspace(3)* %v111 to float addrspace(3)*
  %v199 = getelementptr inbounds float, float addrspace(3)* %v198, i64 %v63
  %v112 = bitcast float addrspace(3)* %v199 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v200 = bitcast i8 addrspace(3)* %v112 to float addrspace(3)*
  store float %v109, float addrspace(3)* %v200, align 4
  br label %bb1
bb26:
  %v113 = add i32 %v49, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v49, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb28
bb27:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb28
bb28:
  %v121 = phi i32 [ %v113, %bb26 ], [ %v49, %bb27 ]
  %v122 = phi i32 [ %v116, %bb26 ], [ %v119, %bb27 ]
  %v123 = phi i32 [ %v117, %bb26 ], [ %v120, %bb27 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb7, label %bb29
bb29:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb6, label %bb2
bb30:
  %v204 = bitcast i8 addrspace(3)* %v58 to float addrspace(3)*
  %v130 = load float, float addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to float*
  %v206 = getelementptr inbounds float, float* %v205, i64 0
  %v132 = bitcast float* %v206 to i8*
  %v207 = bitcast i8* %v132 to float*
  store float %v130, float* %v207, align 4
  br label %bb10
}

declare float @__nv_fmaxf(float, float)

define void @reduce_max_f32_cuda_entry_239487c8da530015(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb26 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb26 ]
  %v17 = add i64 %v82, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to float*
  %v150 = getelementptr inbounds float, float* %v149, i64 %v37
  %v41 = bitcast float* %v150 to i8*
  %v151 = bitcast i8* %v41 to float*
  %v42 = load float, float* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to float addrspace(3)*
  %v156 = getelementptr inbounds float, float addrspace(3)* %v155, i64 %v63
  %v46 = bitcast float addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v121, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb28, label %bb27
bb6:
  call void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_max_f32_cuda_entry_239487c8da53001520reduce_workspace_maxINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB8_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v160 = bitcast i8 addrspace(3)* %v57 to float addrspace(3)*
  %v161 = getelementptr inbounds float, float addrspace(3)* %v160, i64 0
  %v58 = bitcast float addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb31
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x float] addrspace(3)* @__shared_mem_5 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v163, align 8
  %v63 = zext i32 %v14 to i64
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v165 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v164, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v165 to i8*
  %v166 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v166, align 8
  %v167 = bitcast i8 addrspace(3)* %v65 to float addrspace(3)*
  %v168 = getelementptr inbounds float, float addrspace(3)* %v167, i64 %v63
  %v66 = bitcast float addrspace(3)* %v168 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v169 = bitcast i8 addrspace(3)* %v66 to float addrspace(3)*
  store float -999999984306749400.0, float addrspace(3)* %v169, align 4
  %v67 = trunc i64 %v63 to i32
  %v170 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v171 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v170, i32 0, i32 3
  %v68 = bitcast i32* %v171 to i8*
  %v172 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v172, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v174 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, align 8
  br label %bb15
bb15:
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v176 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v176 to i8*
  %v177 = bitcast i8* %v75 to { i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v177, i32 0, i32 0
  %v76 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 1
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 1
  %v81 = bitcast i64* %v186 to i8*
  %v187 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v187, align 8
  %v188 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v189 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v188, i32 0, i32 2
  %v83 = bitcast i1* %v189 to i8*
  %v190 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v190, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v194 = bitcast i8 addrspace(3)* %v46 to float addrspace(3)*
  %v108 = load float, float addrspace(3)* %v194, align 4
  %v109 = call float @__nv_fmaxf(float %v108, float %v42) #0
  br label %bb25
bb25:
  %v195 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v196 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v195, i32 0, i32 0
  %v110 = bitcast i8 addrspace(3)** %v196 to i8*
  %v197 = bitcast i8* %v110 to i8 addrspace(3)**
  %v111 = load i8 addrspace(3)*, i8 addrspace(3)** %v197, align 8
  %v198 = bitcast i8 addrspace(3)* %v111 to float addrspace(3)*
  %v199 = getelementptr inbounds float, float addrspace(3)* %v198, i64 %v63
  %v112 = bitcast float addrspace(3)* %v199 to i8 addrspace(3)*
  br label %bb26
bb26:
  %v200 = bitcast i8 addrspace(3)* %v112 to float addrspace(3)*
  store float %v109, float addrspace(3)* %v200, align 4
  br label %bb1
bb27:
  %v113 = add i32 %v49, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v49, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb29
bb28:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb29
bb29:
  %v121 = phi i32 [ %v113, %bb27 ], [ %v49, %bb28 ]
  %v122 = phi i32 [ %v116, %bb27 ], [ %v119, %bb28 ]
  %v123 = phi i32 [ %v117, %bb27 ], [ %v120, %bb28 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb7, label %bb30
bb30:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb6, label %bb2
bb31:
  %v204 = bitcast i8 addrspace(3)* %v58 to float addrspace(3)*
  %v130 = load float, float addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to float*
  %v206 = getelementptr inbounds float, float* %v205, i64 0
  %v132 = bitcast float* %v206 to i8*
  %v207 = bitcast i8* %v132 to float*
  store float %v130, float* %v207, align 4
  br label %bb10
}

define void @reduce_min_u32_cuda_entry_0e2d7d8e8e928016(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v136 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v136 to i8*
  %v137 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v137 to i8*
  %v14 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  br label %bb1
bb1:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb13
bb2:
  %v16 = phi i32 [ %v81, %bb17 ], [ %v98, %bb27 ]
  %v17 = phi i32 [ %v84, %bb17 ], [ %v99, %bb27 ]
  %v18 = add i64 %v86, 1
  %v138 = alloca i64, align 8
  %v19 = bitcast i64* %v138 to i8*
  %v139 = bitcast i8* %v19 to i64*
  store i64 %v18, i64* %v139, align 8
  %v140 = bitcast i8* %v19 to { i64 }*
  %v20 = load { i64 }, { i64 }* %v140, align 8
  %v21 = extractvalue { i64 } %v20, 0
  %v22 = sub i64 %v21, 0
  %v23 = icmp ule i64 %v22, 0
  %v24 = add i64 %v22, 0
  %v25 = select i1 %v23, i64 %v24, i64 1
  %v26 = icmp eq i64 %v25, 1
  %v141 = alloca { i64 }, align 8
  %v27 = bitcast { i64 }* %v141 to i8*
  %v142 = bitcast i8* %v27 to { i64 }*
  store { i64 } %v20, { i64 }* %v142, align 8
  %v28 = getelementptr inbounds i8, i8* %v27, i64 0
  %v143 = bitcast i8* %v28 to { { i64 } }*
  %v29 = load { { i64 } }, { { i64 } }* %v143, align 8
  %v144 = alloca { { i64 } }, align 8
  %v30 = bitcast { { i64 } }* %v144 to i8*
  %v145 = bitcast i8* %v30 to { { i64 } }*
  store { { i64 } } %v29, { { i64 } }* %v145, align 8
  %v146 = bitcast i8* %v30 to i64*
  %v31 = load i64, i64* %v146, align 8
  %v32 = icmp ugt i64 %v31, 4294967295
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb23, label %bb22
bb3:
  unreachable
bb4:
  %v34 = extractvalue { i32, i32 } %v103, 1
  %v147 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 7
  %v35 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v35 to i32*
  %v36 = load i32, i32* %v149, align 4
  %v37 = mul i32 %v34, %v36
  %v38 = zext i32 %v37 to i64
  %v39 = extractvalue { i8*, i64 } %v10, 1
  %v40 = icmp ult i64 %v38, %v39
  %v41 = extractvalue { i8*, i64 } %v10, 0
  %v150 = bitcast i8* %v41 to i32*
  %v151 = getelementptr inbounds i32, i32* %v150, i64 %v38
  %v42 = bitcast i32* %v151 to i8*
  %v152 = bitcast i8* %v42 to i32*
  %v43 = load i32, i32* %v152, align 4
  %v153 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v154 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v153, i32 0, i32 0
  %v44 = bitcast i8 addrspace(3)** %v154 to i8*
  %v155 = bitcast i8* %v44 to i8 addrspace(3)**
  %v45 = load i8 addrspace(3)*, i8 addrspace(3)** %v155, align 8
  %v46 = getelementptr i8, i8 addrspace(3)* %v45, i64 0
  %v156 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v157 = getelementptr inbounds i32, i32 addrspace(3)* %v156, i64 %v67
  %v47 = bitcast i32 addrspace(3)* %v157 to i8 addrspace(3)*
  br label %bb26
bb5:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb25
bb6:
  %v158 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v159 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v158, i32 0, i32 0
  %v49 = bitcast i8 addrspace(3)** %v159 to i8*
  %v160 = bitcast i8* %v49 to i8 addrspace(3)**
  %v50 = load i8 addrspace(3)*, i8 addrspace(3)** %v160, align 8
  %v161 = bitcast i8 addrspace(3)* %v50 to i32 addrspace(3)*
  %v162 = getelementptr inbounds i32, i32 addrspace(3)* %v161, i64 %v67
  %v51 = bitcast i32 addrspace(3)* %v162 to i8 addrspace(3)*
  br label %bb27
bb7:
  %v52 = phi i64 [ %v58, %bb10 ], [ 64, %bb25 ]
  %v53 = phi i32 [ %v122, %bb10 ], [ 0, %bb25 ]
  %v54 = icmp ult i32 %v53, 7
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb29, label %bb28
bb8:
  call void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_min_u32_cuda_entry_0e2d7d8e8e92801620reduce_workspace_minINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB8_(i64 %v67, i64 %v52, i8* %v12) #0
  br label %bb10
bb9:
  %v57 = icmp eq i32 %v15, 0
  br i1 %v57, label %bb11, label %bb12
bb10:
  %v58 = udiv i64 %v52, 2
  br label %bb7
bb11:
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v164 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v163, i32 0, i32 0
  %v59 = bitcast i8 addrspace(3)** %v164 to i8*
  %v165 = bitcast i8* %v59 to i8 addrspace(3)**
  %v60 = load i8 addrspace(3)*, i8 addrspace(3)** %v165, align 8
  %v61 = getelementptr i8, i8 addrspace(3)* %v60, i64 0
  %v166 = bitcast i8 addrspace(3)* %v61 to i32 addrspace(3)*
  %v167 = getelementptr inbounds i32, i32 addrspace(3)* %v166, i64 0
  %v62 = bitcast i32 addrspace(3)* %v167 to i8 addrspace(3)*
  br label %bb32
bb12:
  ret void
bb13:
  %v63 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb14
bb14:
  %v64 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb15
bb15:
  %v65 = bitcast [128 x i32] addrspace(3)* @__shared_mem_6 to i8 addrspace(3)*
  %v66 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v65, 0
  %v169 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v66, { i8 addrspace(3)* }* %v169, align 8
  %v67 = zext i32 %v15 to i64
  %v170 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v171 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v170, i32 0, i32 0
  %v68 = bitcast i8 addrspace(3)** %v171 to i8*
  %v172 = bitcast i8* %v68 to i8 addrspace(3)**
  %v69 = load i8 addrspace(3)*, i8 addrspace(3)** %v172, align 8
  %v173 = bitcast i8 addrspace(3)* %v69 to i32 addrspace(3)*
  %v174 = getelementptr inbounds i32, i32 addrspace(3)* %v173, i64 %v67
  %v70 = bitcast i32 addrspace(3)* %v174 to i8 addrspace(3)*
  br label %bb16
bb16:
  %v175 = bitcast i8 addrspace(3)* %v70 to i32 addrspace(3)*
  store i32 4294967295, i32 addrspace(3)* %v175, align 4
  %v71 = trunc i64 %v67 to i32
  %v176 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v177 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v176, i32 0, i32 3
  %v72 = bitcast i32* %v177 to i8*
  %v178 = bitcast i8* %v72 to i32*
  %v73 = load i32, i32* %v178, align 4
  %v74 = insertvalue { i32, i32 } undef, i32 %v71, 0
  %v75 = insertvalue { i32, i32 } %v74, i32 %v73, 1
  %v76 = extractvalue { i32, i32 } %v75, 0
  %v77 = extractvalue { i32, i32 } %v75, 1
  %v78 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v76, i32 %v77, i64 128) #0
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v78, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, align 8
  br label %bb17
bb17:
  %v181 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v182 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v181, i32 0, i32 0
  %v79 = bitcast { i32, i32 }* %v182 to i8*
  %v183 = bitcast i8* %v79 to { i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v183, i32 0, i32 0
  %v80 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v185, align 4
  %v186 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v187 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v186, i32 0, i32 0
  %v82 = bitcast { i32, i32 }* %v187 to i8*
  %v188 = bitcast i8* %v82 to { i32, i32 }*
  %v189 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v188, i32 0, i32 1
  %v83 = bitcast i32* %v189 to i8*
  %v190 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v190, align 4
  %v191 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v192 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v191, i32 0, i32 1
  %v85 = bitcast i64* %v192 to i8*
  %v193 = bitcast i8* %v85 to i64*
  %v86 = load i64, i64* %v193, align 8
  %v194 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v195 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v194, i32 0, i32 2
  %v87 = bitcast i1* %v195 to i8*
  %v196 = bitcast i8* %v87 to i1*
  %v88 = load i1, i1* %v196, align 1
  br label %bb2
bb18:
  %v89 = add i32 %v16, %v109
  %v90 = sub i32 %v17, 1
  %v91 = insertvalue { i32, i32 } undef, i32 1, 0
  %v92 = insertvalue { i32, i32 } %v91, i32 %v16, 1
  %v93 = extractvalue { i32, i32 } %v92, 0
  %v94 = extractvalue { i32, i32 } %v92, 1
  br label %bb20
bb19:
  %v95 = insertvalue { i32, i32 } undef, i32 0, 0
  %v96 = extractvalue { i32, i32 } %v95, 0
  %v97 = extractvalue { i32, i32 } %v95, 1
  br label %bb20
bb20:
  %v98 = phi i32 [ %v89, %bb18 ], [ %v16, %bb19 ]
  %v99 = phi i32 [ %v90, %bb18 ], [ %v17, %bb19 ]
  %v100 = phi i32 [ %v93, %bb18 ], [ %v96, %bb19 ]
  %v101 = phi i32 [ %v94, %bb18 ], [ %v97, %bb19 ]
  %v102 = insertvalue { i32, i32 } undef, i32 %v100, 0
  %v103 = insertvalue { i32, i32 } %v102, i32 %v101, 1
  %v104 = extractvalue { i32, i32 } %v103, 0
  %v105 = zext i32 %v104 to i64
  %v106 = icmp eq i64 %v105, 0
  br i1 %v106, label %bb5, label %bb21
bb21:
  %v107 = icmp eq i64 %v105, 1
  br i1 %v107, label %bb4, label %bb3
bb22:
  br label %bb24
bb23:
  %v108 = trunc i64 %v31 to i32
  br label %bb24
bb24:
  %v109 = phi i32 [ 4294967295, %bb22 ], [ %v108, %bb23 ]
  %v110 = icmp ugt i32 %v17, 0
  %v111 = xor i1 %v110, 1
  br i1 %v111, label %bb19, label %bb18
bb25:
  br label %bb7
bb26:
  %v200 = bitcast i8 addrspace(3)* %v47 to i32 addrspace(3)*
  %v112 = load i32, i32 addrspace(3)* %v200, align 4
  %v113 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCslfDnHtpJyg4_13vortx_shaders(i32 %v112, i32 %v43) #0
  br label %bb6
bb27:
  %v201 = bitcast i8 addrspace(3)* %v51 to i32 addrspace(3)*
  store i32 %v113, i32 addrspace(3)* %v201, align 4
  br label %bb2
bb28:
  %v114 = add i32 %v53, 1
  %v115 = insertvalue { i32, i32 } undef, i32 1, 0
  %v116 = insertvalue { i32, i32 } %v115, i32 %v53, 1
  %v117 = extractvalue { i32, i32 } %v116, 0
  %v118 = extractvalue { i32, i32 } %v116, 1
  br label %bb30
bb29:
  %v119 = insertvalue { i32, i32 } undef, i32 0, 0
  %v120 = extractvalue { i32, i32 } %v119, 0
  %v121 = extractvalue { i32, i32 } %v119, 1
  br label %bb30
bb30:
  %v122 = phi i32 [ %v114, %bb28 ], [ %v53, %bb29 ]
  %v123 = phi i32 [ %v117, %bb28 ], [ %v120, %bb29 ]
  %v124 = phi i32 [ %v118, %bb28 ], [ %v121, %bb29 ]
  %v125 = insertvalue { i32, i32 } undef, i32 %v123, 0
  %v126 = insertvalue { i32, i32 } %v125, i32 %v124, 1
  %v127 = extractvalue { i32, i32 } %v126, 0
  %v128 = zext i32 %v127 to i64
  %v129 = icmp eq i64 %v128, 0
  br i1 %v129, label %bb9, label %bb31
bb31:
  %v130 = icmp eq i64 %v128, 1
  br i1 %v130, label %bb8, label %bb3
bb32:
  %v205 = bitcast i8 addrspace(3)* %v62 to i32 addrspace(3)*
  %v131 = load i32, i32 addrspace(3)* %v205, align 4
  %v132 = extractvalue { i8*, i64 } %v11, 0
  %v206 = bitcast i8* %v132 to i32*
  %v207 = getelementptr inbounds i32, i32* %v206, i64 0
  %v133 = bitcast i32* %v207 to i8*
  %v208 = bitcast i8* %v133 to i32*
  store i32 %v131, i32* %v208, align 4
  br label %bb12
}

define void @reduce_mul_f32_cuda_entry_e9d98ad8207eae6b(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb25 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb25 ]
  %v17 = add i64 %v82, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to float*
  %v150 = getelementptr inbounds float, float* %v149, i64 %v37
  %v41 = bitcast float* %v150 to i8*
  %v151 = bitcast i8* %v41 to float*
  %v42 = load float, float* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to float addrspace(3)*
  %v156 = getelementptr inbounds float, float addrspace(3)* %v155, i64 %v63
  %v46 = bitcast float addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v121, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb27, label %bb26
bb6:
  call void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_mulfINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB6_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v160 = bitcast i8 addrspace(3)* %v57 to float addrspace(3)*
  %v161 = getelementptr inbounds float, float addrspace(3)* %v160, i64 0
  %v58 = bitcast float addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb30
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x float] addrspace(3)* @__shared_mem_7 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v163, align 8
  %v63 = zext i32 %v14 to i64
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v165 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v164, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v165 to i8*
  %v166 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v166, align 8
  %v167 = bitcast i8 addrspace(3)* %v65 to float addrspace(3)*
  %v168 = getelementptr inbounds float, float addrspace(3)* %v167, i64 %v63
  %v66 = bitcast float addrspace(3)* %v168 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v169 = bitcast i8 addrspace(3)* %v66 to float addrspace(3)*
  store float 1.0, float addrspace(3)* %v169, align 4
  %v67 = trunc i64 %v63 to i32
  %v170 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v171 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v170, i32 0, i32 3
  %v68 = bitcast i32* %v171 to i8*
  %v172 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v172, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v174 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, align 8
  br label %bb15
bb15:
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v176 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v176 to i8*
  %v177 = bitcast i8* %v75 to { i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v177, i32 0, i32 0
  %v76 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 1
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 1
  %v81 = bitcast i64* %v186 to i8*
  %v187 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v187, align 8
  %v188 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v189 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v188, i32 0, i32 2
  %v83 = bitcast i1* %v189 to i8*
  %v190 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v190, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v194 = bitcast i8 addrspace(3)* %v46 to float addrspace(3)*
  %v108 = load float, float addrspace(3)* %v194, align 4
  %v109 = fmul contract float %v108, %v42
  %v195 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v196 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v195, i32 0, i32 0
  %v110 = bitcast i8 addrspace(3)** %v196 to i8*
  %v197 = bitcast i8* %v110 to i8 addrspace(3)**
  %v111 = load i8 addrspace(3)*, i8 addrspace(3)** %v197, align 8
  %v198 = bitcast i8 addrspace(3)* %v111 to float addrspace(3)*
  %v199 = getelementptr inbounds float, float addrspace(3)* %v198, i64 %v63
  %v112 = bitcast float addrspace(3)* %v199 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v200 = bitcast i8 addrspace(3)* %v112 to float addrspace(3)*
  store float %v109, float addrspace(3)* %v200, align 4
  br label %bb1
bb26:
  %v113 = add i32 %v49, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v49, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb28
bb27:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb28
bb28:
  %v121 = phi i32 [ %v113, %bb26 ], [ %v49, %bb27 ]
  %v122 = phi i32 [ %v116, %bb26 ], [ %v119, %bb27 ]
  %v123 = phi i32 [ %v117, %bb26 ], [ %v120, %bb27 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb7, label %bb29
bb29:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb6, label %bb2
bb30:
  %v204 = bitcast i8 addrspace(3)* %v58 to float addrspace(3)*
  %v130 = load float, float addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to float*
  %v206 = getelementptr inbounds float, float* %v205, i64 0
  %v132 = bitcast float* %v206 to i8*
  %v207 = bitcast i8* %v132 to float*
  store float %v130, float* %v207, align 4
  br label %bb10
}

define void @reduce_sq_norm_cuda_entry_b6de3497e124bb58(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v136 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v136 to i8*
  %v137 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v137 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb25 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb25 ]
  %v17 = add i64 %v82, 1
  %v138 = alloca i64, align 8
  %v18 = bitcast i64* %v138 to i8*
  %v139 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v139, align 8
  %v140 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v140, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v141 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v141 to i8*
  %v142 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v142, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v143 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v143, align 8
  %v144 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v144 to i8*
  %v145 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v145, align 8
  %v146 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v146, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v147 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 7
  %v34 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v149, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v150 = bitcast i8* %v40 to float*
  %v151 = getelementptr inbounds float, float* %v150, i64 %v37
  %v41 = bitcast float* %v151 to i8*
  %v152 = bitcast i8* %v41 to float*
  %v42 = load float, float* %v152, align 4
  %v153 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v154 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v153, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v154 to i8*
  %v155 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v155, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v156 = bitcast i8 addrspace(3)* %v45 to float addrspace(3)*
  %v157 = getelementptr inbounds float, float addrspace(3)* %v156, i64 %v63
  %v46 = bitcast float addrspace(3)* %v157 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v122, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb27, label %bb26
bb6:
  call void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_sumfINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB6_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v158 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v159 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v158, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v159 to i8*
  %v160 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v160, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v161 = bitcast i8 addrspace(3)* %v57 to float addrspace(3)*
  %v162 = getelementptr inbounds float, float addrspace(3)* %v161, i64 0
  %v58 = bitcast float addrspace(3)* %v162 to i8 addrspace(3)*
  br label %bb30
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x float] addrspace(3)* @__shared_mem_8 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v164, align 8
  %v63 = zext i32 %v14 to i64
  %v165 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v166 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v165, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v166 to i8*
  %v167 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v167, align 8
  %v168 = bitcast i8 addrspace(3)* %v65 to float addrspace(3)*
  %v169 = getelementptr inbounds float, float addrspace(3)* %v168, i64 %v63
  %v66 = bitcast float addrspace(3)* %v169 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v170 = bitcast i8 addrspace(3)* %v66 to float addrspace(3)*
  store float 0.0, float addrspace(3)* %v170, align 4
  %v67 = trunc i64 %v63 to i32
  %v171 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v171, i32 0, i32 3
  %v68 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v173, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, align 8
  br label %bb15
bb15:
  %v176 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v177 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v176, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v177 to i8*
  %v178 = bitcast i8* %v75 to { i32, i32 }*
  %v179 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v178, i32 0, i32 0
  %v76 = bitcast i32* %v179 to i8*
  %v180 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v180, align 4
  %v181 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v182 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v181, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v182 to i8*
  %v183 = bitcast i8* %v78 to { i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v183, i32 0, i32 1
  %v79 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v185, align 4
  %v186 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v187 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v186, i32 0, i32 1
  %v81 = bitcast i64* %v187 to i8*
  %v188 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v188, align 8
  %v189 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v190 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v189, i32 0, i32 2
  %v83 = bitcast i1* %v190 to i8*
  %v191 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v191, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v195 = bitcast i8 addrspace(3)* %v46 to float addrspace(3)*
  %v108 = load float, float addrspace(3)* %v195, align 4
  %v109 = fmul contract float %v42, %v42
  %v110 = fadd contract float %v108, %v109
  %v196 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v197 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v196, i32 0, i32 0
  %v111 = bitcast i8 addrspace(3)** %v197 to i8*
  %v198 = bitcast i8* %v111 to i8 addrspace(3)**
  %v112 = load i8 addrspace(3)*, i8 addrspace(3)** %v198, align 8
  %v199 = bitcast i8 addrspace(3)* %v112 to float addrspace(3)*
  %v200 = getelementptr inbounds float, float addrspace(3)* %v199, i64 %v63
  %v113 = bitcast float addrspace(3)* %v200 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v201 = bitcast i8 addrspace(3)* %v113 to float addrspace(3)*
  store float %v110, float addrspace(3)* %v201, align 4
  br label %bb1
bb26:
  %v114 = add i32 %v49, 1
  %v115 = insertvalue { i32, i32 } undef, i32 1, 0
  %v116 = insertvalue { i32, i32 } %v115, i32 %v49, 1
  %v117 = extractvalue { i32, i32 } %v116, 0
  %v118 = extractvalue { i32, i32 } %v116, 1
  br label %bb28
bb27:
  %v119 = insertvalue { i32, i32 } undef, i32 0, 0
  %v120 = extractvalue { i32, i32 } %v119, 0
  %v121 = extractvalue { i32, i32 } %v119, 1
  br label %bb28
bb28:
  %v122 = phi i32 [ %v114, %bb26 ], [ %v49, %bb27 ]
  %v123 = phi i32 [ %v117, %bb26 ], [ %v120, %bb27 ]
  %v124 = phi i32 [ %v118, %bb26 ], [ %v121, %bb27 ]
  %v125 = insertvalue { i32, i32 } undef, i32 %v123, 0
  %v126 = insertvalue { i32, i32 } %v125, i32 %v124, 1
  %v127 = extractvalue { i32, i32 } %v126, 0
  %v128 = zext i32 %v127 to i64
  %v129 = icmp eq i64 %v128, 0
  br i1 %v129, label %bb7, label %bb29
bb29:
  %v130 = icmp eq i64 %v128, 1
  br i1 %v130, label %bb6, label %bb2
bb30:
  %v205 = bitcast i8 addrspace(3)* %v58 to float addrspace(3)*
  %v131 = load float, float addrspace(3)* %v205, align 4
  %v132 = extractvalue { i8*, i64 } %v11, 0
  %v206 = bitcast i8* %v132 to float*
  %v207 = getelementptr inbounds float, float* %v206, i64 0
  %v133 = bitcast float* %v207 to i8*
  %v208 = bitcast i8* %v133 to float*
  store float %v131, float* %v208, align 4
  br label %bb10
}

define void @reduce_add_i32_cuda_entry_23d17cc9dc93c126(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb11
bb1:
  %v15 = phi i32 [ %v77, %bb15 ], [ %v94, %bb25 ]
  %v16 = phi i32 [ %v80, %bb15 ], [ %v95, %bb25 ]
  %v17 = add i64 %v82, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb21, label %bb20
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v99, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to i32*
  %v150 = getelementptr inbounds i32, i32* %v149, i64 %v37
  %v41 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v41 to i32*
  %v42 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to i32 addrspace(3)*
  %v156 = getelementptr inbounds i32, i32 addrspace(3)* %v155, i64 %v63
  %v46 = bitcast i32 addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb24
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb23
bb5:
  %v48 = phi i64 [ %v54, %bb8 ], [ 64, %bb23 ]
  %v49 = phi i32 [ %v121, %bb8 ], [ 0, %bb23 ]
  %v50 = icmp ult i32 %v49, 7
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb27, label %bb26
bb6:
  call void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_sumlINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB6_(i64 %v63, i64 %v48, i8* %v12) #0
  br label %bb8
bb7:
  %v53 = icmp eq i32 %v14, 0
  br i1 %v53, label %bb9, label %bb10
bb8:
  %v54 = udiv i64 %v48, 2
  br label %bb5
bb9:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v55 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v55 to i8 addrspace(3)**
  %v56 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v57 = getelementptr i8, i8 addrspace(3)* %v56, i64 0
  %v160 = bitcast i8 addrspace(3)* %v57 to i32 addrspace(3)*
  %v161 = getelementptr inbounds i32, i32 addrspace(3)* %v160, i64 0
  %v58 = bitcast i32 addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb30
bb10:
  ret void
bb11:
  %v59 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb12
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb13
bb13:
  %v61 = bitcast [128 x i32] addrspace(3)* @__shared_mem_9 to i8 addrspace(3)*
  %v62 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v61, 0
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v62, { i8 addrspace(3)* }* %v163, align 8
  %v63 = zext i32 %v14 to i64
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v165 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v164, i32 0, i32 0
  %v64 = bitcast i8 addrspace(3)** %v165 to i8*
  %v166 = bitcast i8* %v64 to i8 addrspace(3)**
  %v65 = load i8 addrspace(3)*, i8 addrspace(3)** %v166, align 8
  %v167 = bitcast i8 addrspace(3)* %v65 to i32 addrspace(3)*
  %v168 = getelementptr inbounds i32, i32 addrspace(3)* %v167, i64 %v63
  %v66 = bitcast i32 addrspace(3)* %v168 to i8 addrspace(3)*
  br label %bb14
bb14:
  %v169 = bitcast i8 addrspace(3)* %v66 to i32 addrspace(3)*
  store i32 0, i32 addrspace(3)* %v169, align 4
  %v67 = trunc i64 %v63 to i32
  %v170 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v171 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v170, i32 0, i32 3
  %v68 = bitcast i32* %v171 to i8*
  %v172 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v172, align 4
  %v70 = insertvalue { i32, i32 } undef, i32 %v67, 0
  %v71 = insertvalue { i32, i32 } %v70, i32 %v69, 1
  %v72 = extractvalue { i32, i32 } %v71, 0
  %v73 = extractvalue { i32, i32 } %v71, 1
  %v74 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v72, i32 %v73, i64 128) #0
  %v174 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v74, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, align 8
  br label %bb15
bb15:
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v176 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, i32 0, i32 0
  %v75 = bitcast { i32, i32 }* %v176 to i8*
  %v177 = bitcast i8* %v75 to { i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v177, i32 0, i32 0
  %v76 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 1
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 1
  %v81 = bitcast i64* %v186 to i8*
  %v187 = bitcast i8* %v81 to i64*
  %v82 = load i64, i64* %v187, align 8
  %v188 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v189 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v188, i32 0, i32 2
  %v83 = bitcast i1* %v189 to i8*
  %v190 = bitcast i8* %v83 to i1*
  %v84 = load i1, i1* %v190, align 1
  br label %bb1
bb16:
  %v85 = add i32 %v15, %v105
  %v86 = sub i32 %v16, 1
  %v87 = insertvalue { i32, i32 } undef, i32 1, 0
  %v88 = insertvalue { i32, i32 } %v87, i32 %v15, 1
  %v89 = extractvalue { i32, i32 } %v88, 0
  %v90 = extractvalue { i32, i32 } %v88, 1
  br label %bb18
bb17:
  %v91 = insertvalue { i32, i32 } undef, i32 0, 0
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb18
bb18:
  %v94 = phi i32 [ %v85, %bb16 ], [ %v15, %bb17 ]
  %v95 = phi i32 [ %v86, %bb16 ], [ %v16, %bb17 ]
  %v96 = phi i32 [ %v89, %bb16 ], [ %v92, %bb17 ]
  %v97 = phi i32 [ %v90, %bb16 ], [ %v93, %bb17 ]
  %v98 = insertvalue { i32, i32 } undef, i32 %v96, 0
  %v99 = insertvalue { i32, i32 } %v98, i32 %v97, 1
  %v100 = extractvalue { i32, i32 } %v99, 0
  %v101 = zext i32 %v100 to i64
  %v102 = icmp eq i64 %v101, 0
  br i1 %v102, label %bb4, label %bb19
bb19:
  %v103 = icmp eq i64 %v101, 1
  br i1 %v103, label %bb3, label %bb2
bb20:
  br label %bb22
bb21:
  %v104 = trunc i64 %v30 to i32
  br label %bb22
bb22:
  %v105 = phi i32 [ 4294967295, %bb20 ], [ %v104, %bb21 ]
  %v106 = icmp ugt i32 %v16, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb17, label %bb16
bb23:
  br label %bb5
bb24:
  %v194 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v108 = load i32, i32 addrspace(3)* %v194, align 4
  %v109 = add i32 %v108, %v42
  %v195 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v196 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v195, i32 0, i32 0
  %v110 = bitcast i8 addrspace(3)** %v196 to i8*
  %v197 = bitcast i8* %v110 to i8 addrspace(3)**
  %v111 = load i8 addrspace(3)*, i8 addrspace(3)** %v197, align 8
  %v198 = bitcast i8 addrspace(3)* %v111 to i32 addrspace(3)*
  %v199 = getelementptr inbounds i32, i32 addrspace(3)* %v198, i64 %v63
  %v112 = bitcast i32 addrspace(3)* %v199 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v200 = bitcast i8 addrspace(3)* %v112 to i32 addrspace(3)*
  store i32 %v109, i32 addrspace(3)* %v200, align 4
  br label %bb1
bb26:
  %v113 = add i32 %v49, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v49, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb28
bb27:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb28
bb28:
  %v121 = phi i32 [ %v113, %bb26 ], [ %v49, %bb27 ]
  %v122 = phi i32 [ %v116, %bb26 ], [ %v119, %bb27 ]
  %v123 = phi i32 [ %v117, %bb26 ], [ %v120, %bb27 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb7, label %bb29
bb29:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb6, label %bb2
bb30:
  %v204 = bitcast i8 addrspace(3)* %v58 to i32 addrspace(3)*
  %v130 = load i32, i32 addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to i32*
  %v206 = getelementptr inbounds i32, i32* %v205, i64 0
  %v132 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v132 to i32*
  store i32 %v130, i32* %v207, align 4
  br label %bb10
}

define void @reduce_min_i32_cuda_entry_ae0d0e5623f305d7(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v136 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v136 to i8*
  %v137 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v137 to i8*
  %v14 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  br label %bb1
bb1:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb13
bb2:
  %v16 = phi i32 [ %v81, %bb17 ], [ %v98, %bb27 ]
  %v17 = phi i32 [ %v84, %bb17 ], [ %v99, %bb27 ]
  %v18 = add i64 %v86, 1
  %v138 = alloca i64, align 8
  %v19 = bitcast i64* %v138 to i8*
  %v139 = bitcast i8* %v19 to i64*
  store i64 %v18, i64* %v139, align 8
  %v140 = bitcast i8* %v19 to { i64 }*
  %v20 = load { i64 }, { i64 }* %v140, align 8
  %v21 = extractvalue { i64 } %v20, 0
  %v22 = sub i64 %v21, 0
  %v23 = icmp ule i64 %v22, 0
  %v24 = add i64 %v22, 0
  %v25 = select i1 %v23, i64 %v24, i64 1
  %v26 = icmp eq i64 %v25, 1
  %v141 = alloca { i64 }, align 8
  %v27 = bitcast { i64 }* %v141 to i8*
  %v142 = bitcast i8* %v27 to { i64 }*
  store { i64 } %v20, { i64 }* %v142, align 8
  %v28 = getelementptr inbounds i8, i8* %v27, i64 0
  %v143 = bitcast i8* %v28 to { { i64 } }*
  %v29 = load { { i64 } }, { { i64 } }* %v143, align 8
  %v144 = alloca { { i64 } }, align 8
  %v30 = bitcast { { i64 } }* %v144 to i8*
  %v145 = bitcast i8* %v30 to { { i64 } }*
  store { { i64 } } %v29, { { i64 } }* %v145, align 8
  %v146 = bitcast i8* %v30 to i64*
  %v31 = load i64, i64* %v146, align 8
  %v32 = icmp ugt i64 %v31, 4294967295
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb23, label %bb22
bb3:
  unreachable
bb4:
  %v34 = extractvalue { i32, i32 } %v103, 1
  %v147 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 7
  %v35 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v35 to i32*
  %v36 = load i32, i32* %v149, align 4
  %v37 = mul i32 %v34, %v36
  %v38 = zext i32 %v37 to i64
  %v39 = extractvalue { i8*, i64 } %v10, 1
  %v40 = icmp ult i64 %v38, %v39
  %v41 = extractvalue { i8*, i64 } %v10, 0
  %v150 = bitcast i8* %v41 to i32*
  %v151 = getelementptr inbounds i32, i32* %v150, i64 %v38
  %v42 = bitcast i32* %v151 to i8*
  %v152 = bitcast i8* %v42 to i32*
  %v43 = load i32, i32* %v152, align 4
  %v153 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v154 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v153, i32 0, i32 0
  %v44 = bitcast i8 addrspace(3)** %v154 to i8*
  %v155 = bitcast i8* %v44 to i8 addrspace(3)**
  %v45 = load i8 addrspace(3)*, i8 addrspace(3)** %v155, align 8
  %v46 = getelementptr i8, i8 addrspace(3)* %v45, i64 0
  %v156 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v157 = getelementptr inbounds i32, i32 addrspace(3)* %v156, i64 %v67
  %v47 = bitcast i32 addrspace(3)* %v157 to i8 addrspace(3)*
  br label %bb26
bb5:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb25
bb6:
  %v158 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v159 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v158, i32 0, i32 0
  %v49 = bitcast i8 addrspace(3)** %v159 to i8*
  %v160 = bitcast i8* %v49 to i8 addrspace(3)**
  %v50 = load i8 addrspace(3)*, i8 addrspace(3)** %v160, align 8
  %v161 = bitcast i8 addrspace(3)* %v50 to i32 addrspace(3)*
  %v162 = getelementptr inbounds i32, i32 addrspace(3)* %v161, i64 %v67
  %v51 = bitcast i32 addrspace(3)* %v162 to i8 addrspace(3)*
  br label %bb27
bb7:
  %v52 = phi i64 [ %v58, %bb10 ], [ 64, %bb25 ]
  %v53 = phi i32 [ %v122, %bb10 ], [ 0, %bb25 ]
  %v54 = icmp ult i32 %v53, 7
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb29, label %bb28
bb8:
  call void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_min_i32_cuda_entry_ae0d0e5623f305d720reduce_workspace_minINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB8_(i64 %v67, i64 %v52, i8* %v12) #0
  br label %bb10
bb9:
  %v57 = icmp eq i32 %v15, 0
  br i1 %v57, label %bb11, label %bb12
bb10:
  %v58 = udiv i64 %v52, 2
  br label %bb7
bb11:
  %v163 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v164 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v163, i32 0, i32 0
  %v59 = bitcast i8 addrspace(3)** %v164 to i8*
  %v165 = bitcast i8* %v59 to i8 addrspace(3)**
  %v60 = load i8 addrspace(3)*, i8 addrspace(3)** %v165, align 8
  %v61 = getelementptr i8, i8 addrspace(3)* %v60, i64 0
  %v166 = bitcast i8 addrspace(3)* %v61 to i32 addrspace(3)*
  %v167 = getelementptr inbounds i32, i32 addrspace(3)* %v166, i64 0
  %v62 = bitcast i32 addrspace(3)* %v167 to i8 addrspace(3)*
  br label %bb32
bb12:
  ret void
bb13:
  %v63 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb14
bb14:
  %v64 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb15
bb15:
  %v65 = bitcast [128 x i32] addrspace(3)* @__shared_mem_10 to i8 addrspace(3)*
  %v66 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v65, 0
  %v169 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v66, { i8 addrspace(3)* }* %v169, align 8
  %v67 = zext i32 %v15 to i64
  %v170 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v171 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v170, i32 0, i32 0
  %v68 = bitcast i8 addrspace(3)** %v171 to i8*
  %v172 = bitcast i8* %v68 to i8 addrspace(3)**
  %v69 = load i8 addrspace(3)*, i8 addrspace(3)** %v172, align 8
  %v173 = bitcast i8 addrspace(3)* %v69 to i32 addrspace(3)*
  %v174 = getelementptr inbounds i32, i32 addrspace(3)* %v173, i64 %v67
  %v70 = bitcast i32 addrspace(3)* %v174 to i8 addrspace(3)*
  br label %bb16
bb16:
  %v175 = bitcast i8 addrspace(3)* %v70 to i32 addrspace(3)*
  store i32 2147483647, i32 addrspace(3)* %v175, align 4
  %v71 = trunc i64 %v67 to i32
  %v176 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v177 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v176, i32 0, i32 3
  %v72 = bitcast i32* %v177 to i8*
  %v178 = bitcast i8* %v72 to i32*
  %v73 = load i32, i32* %v178, align 4
  %v74 = insertvalue { i32, i32 } undef, i32 %v71, 0
  %v75 = insertvalue { i32, i32 } %v74, i32 %v73, 1
  %v76 = extractvalue { i32, i32 } %v75, 0
  %v77 = extractvalue { i32, i32 } %v75, 1
  %v78 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v76, i32 %v77, i64 128) #0
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v78, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, align 8
  br label %bb17
bb17:
  %v181 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v182 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v181, i32 0, i32 0
  %v79 = bitcast { i32, i32 }* %v182 to i8*
  %v183 = bitcast i8* %v79 to { i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v183, i32 0, i32 0
  %v80 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v185, align 4
  %v186 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v187 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v186, i32 0, i32 0
  %v82 = bitcast { i32, i32 }* %v187 to i8*
  %v188 = bitcast i8* %v82 to { i32, i32 }*
  %v189 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v188, i32 0, i32 1
  %v83 = bitcast i32* %v189 to i8*
  %v190 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v190, align 4
  %v191 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v192 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v191, i32 0, i32 1
  %v85 = bitcast i64* %v192 to i8*
  %v193 = bitcast i8* %v85 to i64*
  %v86 = load i64, i64* %v193, align 8
  %v194 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v195 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v194, i32 0, i32 2
  %v87 = bitcast i1* %v195 to i8*
  %v196 = bitcast i8* %v87 to i1*
  %v88 = load i1, i1* %v196, align 1
  br label %bb2
bb18:
  %v89 = add i32 %v16, %v109
  %v90 = sub i32 %v17, 1
  %v91 = insertvalue { i32, i32 } undef, i32 1, 0
  %v92 = insertvalue { i32, i32 } %v91, i32 %v16, 1
  %v93 = extractvalue { i32, i32 } %v92, 0
  %v94 = extractvalue { i32, i32 } %v92, 1
  br label %bb20
bb19:
  %v95 = insertvalue { i32, i32 } undef, i32 0, 0
  %v96 = extractvalue { i32, i32 } %v95, 0
  %v97 = extractvalue { i32, i32 } %v95, 1
  br label %bb20
bb20:
  %v98 = phi i32 [ %v89, %bb18 ], [ %v16, %bb19 ]
  %v99 = phi i32 [ %v90, %bb18 ], [ %v17, %bb19 ]
  %v100 = phi i32 [ %v93, %bb18 ], [ %v96, %bb19 ]
  %v101 = phi i32 [ %v94, %bb18 ], [ %v97, %bb19 ]
  %v102 = insertvalue { i32, i32 } undef, i32 %v100, 0
  %v103 = insertvalue { i32, i32 } %v102, i32 %v101, 1
  %v104 = extractvalue { i32, i32 } %v103, 0
  %v105 = zext i32 %v104 to i64
  %v106 = icmp eq i64 %v105, 0
  br i1 %v106, label %bb5, label %bb21
bb21:
  %v107 = icmp eq i64 %v105, 1
  br i1 %v107, label %bb4, label %bb3
bb22:
  br label %bb24
bb23:
  %v108 = trunc i64 %v31 to i32
  br label %bb24
bb24:
  %v109 = phi i32 [ 4294967295, %bb22 ], [ %v108, %bb23 ]
  %v110 = icmp ugt i32 %v17, 0
  %v111 = xor i1 %v110, 1
  br i1 %v111, label %bb19, label %bb18
bb25:
  br label %bb7
bb26:
  %v200 = bitcast i8 addrspace(3)* %v47 to i32 addrspace(3)*
  %v112 = load i32, i32 addrspace(3)* %v200, align 4
  %v113 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCslfDnHtpJyg4_13vortx_shaders(i32 %v112, i32 %v43) #0
  br label %bb6
bb27:
  %v201 = bitcast i8 addrspace(3)* %v51 to i32 addrspace(3)*
  store i32 %v113, i32 addrspace(3)* %v201, align 4
  br label %bb2
bb28:
  %v114 = add i32 %v53, 1
  %v115 = insertvalue { i32, i32 } undef, i32 1, 0
  %v116 = insertvalue { i32, i32 } %v115, i32 %v53, 1
  %v117 = extractvalue { i32, i32 } %v116, 0
  %v118 = extractvalue { i32, i32 } %v116, 1
  br label %bb30
bb29:
  %v119 = insertvalue { i32, i32 } undef, i32 0, 0
  %v120 = extractvalue { i32, i32 } %v119, 0
  %v121 = extractvalue { i32, i32 } %v119, 1
  br label %bb30
bb30:
  %v122 = phi i32 [ %v114, %bb28 ], [ %v53, %bb29 ]
  %v123 = phi i32 [ %v117, %bb28 ], [ %v120, %bb29 ]
  %v124 = phi i32 [ %v118, %bb28 ], [ %v121, %bb29 ]
  %v125 = insertvalue { i32, i32 } undef, i32 %v123, 0
  %v126 = insertvalue { i32, i32 } %v125, i32 %v124, 1
  %v127 = extractvalue { i32, i32 } %v126, 0
  %v128 = zext i32 %v127 to i64
  %v129 = icmp eq i64 %v128, 0
  br i1 %v129, label %bb9, label %bb31
bb31:
  %v130 = icmp eq i64 %v128, 1
  br i1 %v130, label %bb8, label %bb3
bb32:
  %v205 = bitcast i8 addrspace(3)* %v62 to i32 addrspace(3)*
  %v131 = load i32, i32 addrspace(3)* %v205, align 4
  %v132 = extractvalue { i8*, i64 } %v11, 0
  %v206 = bitcast i8* %v132 to i32*
  %v207 = getelementptr inbounds i32, i32* %v206, i64 0
  %v133 = bitcast i32* %v207 to i8*
  %v208 = bitcast i8* %v133 to i32*
  store i32 %v131, i32* %v208, align 4
  br label %bb12
}

declare float @__nv_fminf(float, float)

define void @reduce_min_f32_cuda_entry_a1b9271d7e5a05b2(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v136 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v136 to i8*
  %v137 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v137 to i8*
  %v14 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  br label %bb1
bb1:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb12
bb2:
  %v16 = phi i32 [ %v78, %bb16 ], [ %v95, %bb27 ]
  %v17 = phi i32 [ %v81, %bb16 ], [ %v96, %bb27 ]
  %v18 = add i64 %v83, 1
  %v138 = alloca i64, align 8
  %v19 = bitcast i64* %v138 to i8*
  %v139 = bitcast i8* %v19 to i64*
  store i64 %v18, i64* %v139, align 8
  %v140 = bitcast i8* %v19 to { i64 }*
  %v20 = load { i64 }, { i64 }* %v140, align 8
  %v21 = extractvalue { i64 } %v20, 0
  %v22 = sub i64 %v21, 0
  %v23 = icmp ule i64 %v22, 0
  %v24 = add i64 %v22, 0
  %v25 = select i1 %v23, i64 %v24, i64 1
  %v26 = icmp eq i64 %v25, 1
  %v141 = alloca { i64 }, align 8
  %v27 = bitcast { i64 }* %v141 to i8*
  %v142 = bitcast i8* %v27 to { i64 }*
  store { i64 } %v20, { i64 }* %v142, align 8
  %v28 = getelementptr inbounds i8, i8* %v27, i64 0
  %v143 = bitcast i8* %v28 to { { i64 } }*
  %v29 = load { { i64 } }, { { i64 } }* %v143, align 8
  %v144 = alloca { { i64 } }, align 8
  %v30 = bitcast { { i64 } }* %v144 to i8*
  %v145 = bitcast i8* %v30 to { { i64 } }*
  store { { i64 } } %v29, { { i64 } }* %v145, align 8
  %v146 = bitcast i8* %v30 to i64*
  %v31 = load i64, i64* %v146, align 8
  %v32 = icmp ugt i64 %v31, 4294967295
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb22, label %bb21
bb3:
  unreachable
bb4:
  %v34 = extractvalue { i32, i32 } %v100, 1
  %v147 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 7
  %v35 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v35 to i32*
  %v36 = load i32, i32* %v149, align 4
  %v37 = mul i32 %v34, %v36
  %v38 = zext i32 %v37 to i64
  %v39 = extractvalue { i8*, i64 } %v10, 1
  %v40 = icmp ult i64 %v38, %v39
  %v41 = extractvalue { i8*, i64 } %v10, 0
  %v150 = bitcast i8* %v41 to float*
  %v151 = getelementptr inbounds float, float* %v150, i64 %v38
  %v42 = bitcast float* %v151 to i8*
  %v152 = bitcast i8* %v42 to float*
  %v43 = load float, float* %v152, align 4
  %v153 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v154 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v153, i32 0, i32 0
  %v44 = bitcast i8 addrspace(3)** %v154 to i8*
  %v155 = bitcast i8* %v44 to i8 addrspace(3)**
  %v45 = load i8 addrspace(3)*, i8 addrspace(3)** %v155, align 8
  %v46 = getelementptr i8, i8 addrspace(3)* %v45, i64 0
  %v156 = bitcast i8 addrspace(3)* %v46 to float addrspace(3)*
  %v157 = getelementptr inbounds float, float addrspace(3)* %v156, i64 %v64
  %v47 = bitcast float addrspace(3)* %v157 to i8 addrspace(3)*
  br label %bb25
bb5:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb24
bb6:
  %v49 = phi i64 [ %v55, %bb9 ], [ 64, %bb24 ]
  %v50 = phi i32 [ %v122, %bb9 ], [ 0, %bb24 ]
  %v51 = icmp ult i32 %v50, 7
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb29, label %bb28
bb7:
  call void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_min_f32_cuda_entry_a1b9271d7e5a05b220reduce_workspace_minINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB8_(i64 %v64, i64 %v49, i8* %v12) #0
  br label %bb9
bb8:
  %v54 = icmp eq i32 %v15, 0
  br i1 %v54, label %bb10, label %bb11
bb9:
  %v55 = udiv i64 %v49, 2
  br label %bb6
bb10:
  %v158 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v159 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v158, i32 0, i32 0
  %v56 = bitcast i8 addrspace(3)** %v159 to i8*
  %v160 = bitcast i8* %v56 to i8 addrspace(3)**
  %v57 = load i8 addrspace(3)*, i8 addrspace(3)** %v160, align 8
  %v58 = getelementptr i8, i8 addrspace(3)* %v57, i64 0
  %v161 = bitcast i8 addrspace(3)* %v58 to float addrspace(3)*
  %v162 = getelementptr inbounds float, float addrspace(3)* %v161, i64 0
  %v59 = bitcast float addrspace(3)* %v162 to i8 addrspace(3)*
  br label %bb32
bb11:
  ret void
bb12:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb13
bb13:
  %v61 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb14
bb14:
  %v62 = bitcast [128 x float] addrspace(3)* @__shared_mem_11 to i8 addrspace(3)*
  %v63 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v62, 0
  %v164 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v63, { i8 addrspace(3)* }* %v164, align 8
  %v64 = zext i32 %v15 to i64
  %v165 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v166 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v165, i32 0, i32 0
  %v65 = bitcast i8 addrspace(3)** %v166 to i8*
  %v167 = bitcast i8* %v65 to i8 addrspace(3)**
  %v66 = load i8 addrspace(3)*, i8 addrspace(3)** %v167, align 8
  %v168 = bitcast i8 addrspace(3)* %v66 to float addrspace(3)*
  %v169 = getelementptr inbounds float, float addrspace(3)* %v168, i64 %v64
  %v67 = bitcast float addrspace(3)* %v169 to i8 addrspace(3)*
  br label %bb15
bb15:
  %v170 = bitcast i8 addrspace(3)* %v67 to float addrspace(3)*
  store float 999999984306749400.0, float addrspace(3)* %v170, align 4
  %v68 = trunc i64 %v64 to i32
  %v171 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v171, i32 0, i32 3
  %v69 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v69 to i32*
  %v70 = load i32, i32* %v173, align 4
  %v71 = insertvalue { i32, i32 } undef, i32 %v68, 0
  %v72 = insertvalue { i32, i32 } %v71, i32 %v70, 1
  %v73 = extractvalue { i32, i32 } %v72, 0
  %v74 = extractvalue { i32, i32 } %v72, 1
  %v75 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v73, i32 %v74, i64 128) #0
  %v175 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v75, { { i32, i32 }, i64, i1, [7 x i8] }* %v175, align 8
  br label %bb16
bb16:
  %v176 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v177 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v176, i32 0, i32 0
  %v76 = bitcast { i32, i32 }* %v177 to i8*
  %v178 = bitcast i8* %v76 to { i32, i32 }*
  %v179 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v178, i32 0, i32 0
  %v77 = bitcast i32* %v179 to i8*
  %v180 = bitcast i8* %v77 to i32*
  %v78 = load i32, i32* %v180, align 4
  %v181 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v182 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v181, i32 0, i32 0
  %v79 = bitcast { i32, i32 }* %v182 to i8*
  %v183 = bitcast i8* %v79 to { i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v183, i32 0, i32 1
  %v80 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v185, align 4
  %v186 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v187 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v186, i32 0, i32 1
  %v82 = bitcast i64* %v187 to i8*
  %v188 = bitcast i8* %v82 to i64*
  %v83 = load i64, i64* %v188, align 8
  %v189 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v190 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v189, i32 0, i32 2
  %v84 = bitcast i1* %v190 to i8*
  %v191 = bitcast i8* %v84 to i1*
  %v85 = load i1, i1* %v191, align 1
  br label %bb2
bb17:
  %v86 = add i32 %v16, %v106
  %v87 = sub i32 %v17, 1
  %v88 = insertvalue { i32, i32 } undef, i32 1, 0
  %v89 = insertvalue { i32, i32 } %v88, i32 %v16, 1
  %v90 = extractvalue { i32, i32 } %v89, 0
  %v91 = extractvalue { i32, i32 } %v89, 1
  br label %bb19
bb18:
  %v92 = insertvalue { i32, i32 } undef, i32 0, 0
  %v93 = extractvalue { i32, i32 } %v92, 0
  %v94 = extractvalue { i32, i32 } %v92, 1
  br label %bb19
bb19:
  %v95 = phi i32 [ %v86, %bb17 ], [ %v16, %bb18 ]
  %v96 = phi i32 [ %v87, %bb17 ], [ %v17, %bb18 ]
  %v97 = phi i32 [ %v90, %bb17 ], [ %v93, %bb18 ]
  %v98 = phi i32 [ %v91, %bb17 ], [ %v94, %bb18 ]
  %v99 = insertvalue { i32, i32 } undef, i32 %v97, 0
  %v100 = insertvalue { i32, i32 } %v99, i32 %v98, 1
  %v101 = extractvalue { i32, i32 } %v100, 0
  %v102 = zext i32 %v101 to i64
  %v103 = icmp eq i64 %v102, 0
  br i1 %v103, label %bb5, label %bb20
bb20:
  %v104 = icmp eq i64 %v102, 1
  br i1 %v104, label %bb4, label %bb3
bb21:
  br label %bb23
bb22:
  %v105 = trunc i64 %v31 to i32
  br label %bb23
bb23:
  %v106 = phi i32 [ 4294967295, %bb21 ], [ %v105, %bb22 ]
  %v107 = icmp ugt i32 %v17, 0
  %v108 = xor i1 %v107, 1
  br i1 %v108, label %bb18, label %bb17
bb24:
  br label %bb6
bb25:
  %v195 = bitcast i8 addrspace(3)* %v47 to float addrspace(3)*
  %v109 = load float, float addrspace(3)* %v195, align 4
  %v110 = call float @__nv_fminf(float %v109, float %v43) #0
  br label %bb26
bb26:
  %v196 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v197 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v196, i32 0, i32 0
  %v111 = bitcast i8 addrspace(3)** %v197 to i8*
  %v198 = bitcast i8* %v111 to i8 addrspace(3)**
  %v112 = load i8 addrspace(3)*, i8 addrspace(3)** %v198, align 8
  %v199 = bitcast i8 addrspace(3)* %v112 to float addrspace(3)*
  %v200 = getelementptr inbounds float, float addrspace(3)* %v199, i64 %v64
  %v113 = bitcast float addrspace(3)* %v200 to i8 addrspace(3)*
  br label %bb27
bb27:
  %v201 = bitcast i8 addrspace(3)* %v113 to float addrspace(3)*
  store float %v110, float addrspace(3)* %v201, align 4
  br label %bb2
bb28:
  %v114 = add i32 %v50, 1
  %v115 = insertvalue { i32, i32 } undef, i32 1, 0
  %v116 = insertvalue { i32, i32 } %v115, i32 %v50, 1
  %v117 = extractvalue { i32, i32 } %v116, 0
  %v118 = extractvalue { i32, i32 } %v116, 1
  br label %bb30
bb29:
  %v119 = insertvalue { i32, i32 } undef, i32 0, 0
  %v120 = extractvalue { i32, i32 } %v119, 0
  %v121 = extractvalue { i32, i32 } %v119, 1
  br label %bb30
bb30:
  %v122 = phi i32 [ %v114, %bb28 ], [ %v50, %bb29 ]
  %v123 = phi i32 [ %v117, %bb28 ], [ %v120, %bb29 ]
  %v124 = phi i32 [ %v118, %bb28 ], [ %v121, %bb29 ]
  %v125 = insertvalue { i32, i32 } undef, i32 %v123, 0
  %v126 = insertvalue { i32, i32 } %v125, i32 %v124, 1
  %v127 = extractvalue { i32, i32 } %v126, 0
  %v128 = zext i32 %v127 to i64
  %v129 = icmp eq i64 %v128, 0
  br i1 %v129, label %bb8, label %bb31
bb31:
  %v130 = icmp eq i64 %v128, 1
  br i1 %v130, label %bb7, label %bb3
bb32:
  %v205 = bitcast i8 addrspace(3)* %v59 to float addrspace(3)*
  %v131 = load float, float addrspace(3)* %v205, align 4
  %v132 = extractvalue { i8*, i64 } %v11, 0
  %v206 = bitcast i8* %v132 to float*
  %v207 = getelementptr inbounds float, float* %v206, i64 0
  %v133 = bitcast float* %v207 to i8*
  %v208 = bitcast i8* %v133 to float*
  store float %v131, float* %v208, align 4
  br label %bb11
}

define void @reduce_max_u32_cuda_entry_1f9cd5e87342aa39(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v135 = alloca { i8 addrspace(3)* }, align 8
  %v12 = bitcast { i8 addrspace(3)* }* %v135 to i8*
  %v136 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v13 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v136 to i8*
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb12
bb1:
  %v15 = phi i32 [ %v80, %bb16 ], [ %v97, %bb26 ]
  %v16 = phi i32 [ %v83, %bb16 ], [ %v98, %bb26 ]
  %v17 = add i64 %v85, 1
  %v137 = alloca i64, align 8
  %v18 = bitcast i64* %v137 to i8*
  %v138 = bitcast i8* %v18 to i64*
  store i64 %v17, i64* %v138, align 8
  %v139 = bitcast i8* %v18 to { i64 }*
  %v19 = load { i64 }, { i64 }* %v139, align 8
  %v20 = extractvalue { i64 } %v19, 0
  %v21 = sub i64 %v20, 0
  %v22 = icmp ule i64 %v21, 0
  %v23 = add i64 %v21, 0
  %v24 = select i1 %v22, i64 %v23, i64 1
  %v25 = icmp eq i64 %v24, 1
  %v140 = alloca { i64 }, align 8
  %v26 = bitcast { i64 }* %v140 to i8*
  %v141 = bitcast i8* %v26 to { i64 }*
  store { i64 } %v19, { i64 }* %v141, align 8
  %v27 = getelementptr inbounds i8, i8* %v26, i64 0
  %v142 = bitcast i8* %v27 to { { i64 } }*
  %v28 = load { { i64 } }, { { i64 } }* %v142, align 8
  %v143 = alloca { { i64 } }, align 8
  %v29 = bitcast { { i64 } }* %v143 to i8*
  %v144 = bitcast i8* %v29 to { { i64 } }*
  store { { i64 } } %v28, { { i64 } }* %v144, align 8
  %v145 = bitcast i8* %v29 to i64*
  %v30 = load i64, i64* %v145, align 8
  %v31 = icmp ugt i64 %v30, 4294967295
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb22, label %bb21
bb2:
  unreachable
bb3:
  %v33 = extractvalue { i32, i32 } %v102, 1
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v146, i32 0, i32 7
  %v34 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v148, align 4
  %v36 = mul i32 %v33, %v35
  %v37 = zext i32 %v36 to i64
  %v38 = extractvalue { i8*, i64 } %v10, 1
  %v39 = icmp ult i64 %v37, %v38
  %v40 = extractvalue { i8*, i64 } %v10, 0
  %v149 = bitcast i8* %v40 to i32*
  %v150 = getelementptr inbounds i32, i32* %v149, i64 %v37
  %v41 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v41 to i32*
  %v42 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v153 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v152, i32 0, i32 0
  %v43 = bitcast i8 addrspace(3)** %v153 to i8*
  %v154 = bitcast i8* %v43 to i8 addrspace(3)**
  %v44 = load i8 addrspace(3)*, i8 addrspace(3)** %v154, align 8
  %v45 = getelementptr i8, i8 addrspace(3)* %v44, i64 0
  %v155 = bitcast i8 addrspace(3)* %v45 to i32 addrspace(3)*
  %v156 = getelementptr inbounds i32, i32 addrspace(3)* %v155, i64 %v66
  %v46 = bitcast i32 addrspace(3)* %v156 to i8 addrspace(3)*
  br label %bb25
bb4:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb24
bb5:
  %v157 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v158 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v157, i32 0, i32 0
  %v48 = bitcast i8 addrspace(3)** %v158 to i8*
  %v159 = bitcast i8* %v48 to i8 addrspace(3)**
  %v49 = load i8 addrspace(3)*, i8 addrspace(3)** %v159, align 8
  %v160 = bitcast i8 addrspace(3)* %v49 to i32 addrspace(3)*
  %v161 = getelementptr inbounds i32, i32 addrspace(3)* %v160, i64 %v66
  %v50 = bitcast i32 addrspace(3)* %v161 to i8 addrspace(3)*
  br label %bb26
bb6:
  %v51 = phi i64 [ %v57, %bb9 ], [ 64, %bb24 ]
  %v52 = phi i32 [ %v121, %bb9 ], [ 0, %bb24 ]
  %v53 = icmp ult i32 %v52, 7
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb28, label %bb27
bb7:
  call void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_max_u32_cuda_entry_1f9cd5e87342aa3920reduce_workspace_maxINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB8_(i64 %v66, i64 %v51, i8* %v12) #0
  br label %bb9
bb8:
  %v56 = icmp eq i32 %v14, 0
  br i1 %v56, label %bb10, label %bb11
bb9:
  %v57 = udiv i64 %v51, 2
  br label %bb6
bb10:
  %v162 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v163 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v162, i32 0, i32 0
  %v58 = bitcast i8 addrspace(3)** %v163 to i8*
  %v164 = bitcast i8* %v58 to i8 addrspace(3)**
  %v59 = load i8 addrspace(3)*, i8 addrspace(3)** %v164, align 8
  %v60 = getelementptr i8, i8 addrspace(3)* %v59, i64 0
  %v165 = bitcast i8 addrspace(3)* %v60 to i32 addrspace(3)*
  %v166 = getelementptr inbounds i32, i32 addrspace(3)* %v165, i64 0
  %v61 = bitcast i32 addrspace(3)* %v166 to i8 addrspace(3)*
  br label %bb31
bb11:
  ret void
bb12:
  %v62 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb13
bb13:
  %v63 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb14
bb14:
  %v64 = bitcast [128 x i32] addrspace(3)* @__shared_mem_12 to i8 addrspace(3)*
  %v65 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v64, 0
  %v168 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  store { i8 addrspace(3)* } %v65, { i8 addrspace(3)* }* %v168, align 8
  %v66 = zext i32 %v14 to i64
  %v169 = bitcast i8* %v12 to { i8 addrspace(3)* }*
  %v170 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v169, i32 0, i32 0
  %v67 = bitcast i8 addrspace(3)** %v170 to i8*
  %v171 = bitcast i8* %v67 to i8 addrspace(3)**
  %v68 = load i8 addrspace(3)*, i8 addrspace(3)** %v171, align 8
  %v172 = bitcast i8 addrspace(3)* %v68 to i32 addrspace(3)*
  %v173 = getelementptr inbounds i32, i32 addrspace(3)* %v172, i64 %v66
  %v69 = bitcast i32 addrspace(3)* %v173 to i8 addrspace(3)*
  br label %bb15
bb15:
  %v174 = bitcast i8 addrspace(3)* %v69 to i32 addrspace(3)*
  store i32 0, i32 addrspace(3)* %v174, align 4
  %v70 = trunc i64 %v66 to i32
  %v175 = bitcast i8* %v9 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v176 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v175, i32 0, i32 3
  %v71 = bitcast i32* %v176 to i8*
  %v177 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v177, align 4
  %v73 = insertvalue { i32, i32 } undef, i32 %v70, 0
  %v74 = insertvalue { i32, i32 } %v73, i32 %v72, 1
  %v75 = extractvalue { i32, i32 } %v74, 0
  %v76 = extractvalue { i32, i32 } %v74, 1
  %v77 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v75, i32 %v76, i64 128) #0
  %v179 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v77, { { i32, i32 }, i64, i1, [7 x i8] }* %v179, align 8
  br label %bb16
bb16:
  %v180 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v181 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v180, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v181 to i8*
  %v182 = bitcast i8* %v78 to { i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v182, i32 0, i32 0
  %v79 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v186 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v185, i32 0, i32 0
  %v81 = bitcast { i32, i32 }* %v186 to i8*
  %v187 = bitcast i8* %v81 to { i32, i32 }*
  %v188 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v187, i32 0, i32 1
  %v82 = bitcast i32* %v188 to i8*
  %v189 = bitcast i8* %v82 to i32*
  %v83 = load i32, i32* %v189, align 4
  %v190 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v191 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v190, i32 0, i32 1
  %v84 = bitcast i64* %v191 to i8*
  %v192 = bitcast i8* %v84 to i64*
  %v85 = load i64, i64* %v192, align 8
  %v193 = bitcast i8* %v13 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v194 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v193, i32 0, i32 2
  %v86 = bitcast i1* %v194 to i8*
  %v195 = bitcast i8* %v86 to i1*
  %v87 = load i1, i1* %v195, align 1
  br label %bb1
bb17:
  %v88 = add i32 %v15, %v108
  %v89 = sub i32 %v16, 1
  %v90 = insertvalue { i32, i32 } undef, i32 1, 0
  %v91 = insertvalue { i32, i32 } %v90, i32 %v15, 1
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb19
bb18:
  %v94 = insertvalue { i32, i32 } undef, i32 0, 0
  %v95 = extractvalue { i32, i32 } %v94, 0
  %v96 = extractvalue { i32, i32 } %v94, 1
  br label %bb19
bb19:
  %v97 = phi i32 [ %v88, %bb17 ], [ %v15, %bb18 ]
  %v98 = phi i32 [ %v89, %bb17 ], [ %v16, %bb18 ]
  %v99 = phi i32 [ %v92, %bb17 ], [ %v95, %bb18 ]
  %v100 = phi i32 [ %v93, %bb17 ], [ %v96, %bb18 ]
  %v101 = insertvalue { i32, i32 } undef, i32 %v99, 0
  %v102 = insertvalue { i32, i32 } %v101, i32 %v100, 1
  %v103 = extractvalue { i32, i32 } %v102, 0
  %v104 = zext i32 %v103 to i64
  %v105 = icmp eq i64 %v104, 0
  br i1 %v105, label %bb4, label %bb20
bb20:
  %v106 = icmp eq i64 %v104, 1
  br i1 %v106, label %bb3, label %bb2
bb21:
  br label %bb23
bb22:
  %v107 = trunc i64 %v30 to i32
  br label %bb23
bb23:
  %v108 = phi i32 [ 4294967295, %bb21 ], [ %v107, %bb22 ]
  %v109 = icmp ugt i32 %v16, 0
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb18, label %bb17
bb24:
  br label %bb6
bb25:
  %v199 = bitcast i8 addrspace(3)* %v46 to i32 addrspace(3)*
  %v111 = load i32, i32 addrspace(3)* %v199, align 4
  %v112 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCslfDnHtpJyg4_13vortx_shaders(i32 %v111, i32 %v42) #0
  br label %bb5
bb26:
  %v200 = bitcast i8 addrspace(3)* %v50 to i32 addrspace(3)*
  store i32 %v112, i32 addrspace(3)* %v200, align 4
  br label %bb1
bb27:
  %v113 = add i32 %v52, 1
  %v114 = insertvalue { i32, i32 } undef, i32 1, 0
  %v115 = insertvalue { i32, i32 } %v114, i32 %v52, 1
  %v116 = extractvalue { i32, i32 } %v115, 0
  %v117 = extractvalue { i32, i32 } %v115, 1
  br label %bb29
bb28:
  %v118 = insertvalue { i32, i32 } undef, i32 0, 0
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb29
bb29:
  %v121 = phi i32 [ %v113, %bb27 ], [ %v52, %bb28 ]
  %v122 = phi i32 [ %v116, %bb27 ], [ %v119, %bb28 ]
  %v123 = phi i32 [ %v117, %bb27 ], [ %v120, %bb28 ]
  %v124 = insertvalue { i32, i32 } undef, i32 %v122, 0
  %v125 = insertvalue { i32, i32 } %v124, i32 %v123, 1
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = zext i32 %v126 to i64
  %v128 = icmp eq i64 %v127, 0
  br i1 %v128, label %bb8, label %bb30
bb30:
  %v129 = icmp eq i64 %v127, 1
  br i1 %v129, label %bb7, label %bb2
bb31:
  %v204 = bitcast i8 addrspace(3)* %v61 to i32 addrspace(3)*
  %v130 = load i32, i32 addrspace(3)* %v204, align 4
  %v131 = extractvalue { i8*, i64 } %v11, 0
  %v205 = bitcast i8* %v131 to i32*
  %v206 = getelementptr inbounds i32, i32* %v205, i64 0
  %v132 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v132 to i32*
  store i32 %v130, i32* %v207, align 4
  br label %bb11
}

define void @gpu_add_cuda_entry_3172e2e34d229764(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v140 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v140 to i8*
  %v141 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v15 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v141 to i8*
  %v142 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v142 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v143 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v143, align 4
  br label %bb1
bb1:
  %v144 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v145 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v144, i32 0, i32 0
  %v18 = bitcast i32* %v145 to i8*
  %v146 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v146, align 4
  %v147 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 0
  %v20 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v149, align 4
  %v150 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v151 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v150, i32 0, i32 1
  %v22 = bitcast i32* %v151 to i8*
  %v152 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v152, align 4
  %v24 = mul i32 %v21, %v23
  %v153 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v154 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v153, i32 0, i32 2
  %v25 = bitcast i32* %v154 to i8*
  %v155 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v155, align 4
  %v27 = mul i32 %v24, %v26
  %v156 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v157 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v156, i32 0, i32 3
  %v28 = bitcast i32* %v157 to i8*
  %v158 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v158, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = insertvalue { i32, i32 } undef, i32 %v19, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v30, 1
  %v33 = extractvalue { i32, i32 } %v32, 0
  %v34 = extractvalue { i32, i32 } %v32, 1
  %v35 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v33, i32 %v34, i64 16776960) #0
  %v160 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v35, { { i32, i32 }, i64, i1, [7 x i8] }* %v160, align 8
  br label %bb7
bb2:
  %v36 = phi i32 [ %v124, %bb6 ], [ %v107, %bb7 ]
  %v37 = phi i32 [ %v125, %bb6 ], [ %v110, %bb7 ]
  %v38 = add i64 %v112, 1
  %v161 = alloca i64, align 8
  %v39 = bitcast i64* %v161 to i8*
  %v162 = bitcast i8* %v39 to i64*
  store i64 %v38, i64* %v162, align 8
  %v163 = bitcast i8* %v39 to { i64 }*
  %v40 = load { i64 }, { i64 }* %v163, align 8
  %v41 = extractvalue { i64 } %v40, 0
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  %v164 = alloca { i64 }, align 8
  %v47 = bitcast { i64 }* %v164 to i8*
  %v165 = bitcast i8* %v47 to { i64 }*
  store { i64 } %v40, { i64 }* %v165, align 8
  %v48 = getelementptr inbounds i8, i8* %v47, i64 0
  %v166 = bitcast i8* %v48 to { { i64 } }*
  %v49 = load { { i64 } }, { { i64 } }* %v166, align 8
  %v167 = alloca { { i64 } }, align 8
  %v50 = bitcast { { i64 } }* %v167 to i8*
  %v168 = bitcast i8* %v50 to { { i64 } }*
  store { { i64 } } %v49, { { i64 } }* %v168, align 8
  %v169 = bitcast i8* %v50 to i64*
  %v51 = load i64, i64* %v169, align 8
  %v52 = icmp ugt i64 %v51, 4294967295
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v54 = extractvalue { i32, i32 } %v129, 1
  %v55 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v54) #0
  %v170 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v55, { i32, i32, i32, i32 }* %v170, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v171 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v171, i32 0, i32 0
  %v56 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v173, align 4
  %v174 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v175 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v174, i32 0, i32 1
  %v58 = bitcast i32* %v175 to i8*
  %v176 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v176, align 4
  %v177 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v177, i32 0, i32 2
  %v60 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v181 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v180, i32 0, i32 3
  %v62 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v182, align 4
  %v183 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v183, i32 0, i32 4
  %v64 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v185, align 4
  %v66 = mul i32 %v57, %v65
  %v186 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v187 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v186, i32 0, i32 5
  %v67 = bitcast i32* %v187 to i8*
  %v188 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v188, align 4
  %v69 = mul i32 %v59, %v68
  %v70 = add i32 %v66, %v69
  %v189 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v190 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v189, i32 0, i32 6
  %v71 = bitcast i32* %v190 to i8*
  %v191 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v191, align 4
  %v73 = mul i32 %v61, %v72
  %v74 = add i32 %v70, %v73
  %v192 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v193 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v192, i32 0, i32 7
  %v75 = bitcast i32* %v193 to i8*
  %v194 = bitcast i8* %v75 to i32*
  %v76 = load i32, i32* %v194, align 4
  %v77 = mul i32 %v63, %v76
  %v78 = add i32 %v74, %v77
  %v79 = zext i32 %v78 to i64
  %v195 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v196 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v195, i32 0, i32 4
  %v80 = bitcast i32* %v196 to i8*
  %v197 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v197, align 4
  %v82 = mul i32 %v57, %v81
  %v198 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v199 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v198, i32 0, i32 5
  %v83 = bitcast i32* %v199 to i8*
  %v200 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v200, align 4
  %v85 = mul i32 %v59, %v84
  %v86 = add i32 %v82, %v85
  %v201 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v202 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v201, i32 0, i32 6
  %v87 = bitcast i32* %v202 to i8*
  %v203 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v203, align 4
  %v89 = mul i32 %v61, %v88
  %v90 = add i32 %v86, %v89
  %v204 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v205 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v204, i32 0, i32 7
  %v91 = bitcast i32* %v205 to i8*
  %v206 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v206, align 4
  %v93 = mul i32 %v63, %v92
  %v94 = add i32 %v90, %v93
  %v95 = zext i32 %v94 to i64
  %v96 = extractvalue { i8*, i64 } %v13, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { i8*, i64 } %v13, 0
  %v207 = bitcast i8* %v98 to float*
  %v208 = getelementptr inbounds float, float* %v207, i64 %v95
  %v99 = bitcast float* %v208 to i8*
  %v209 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v209, align 4
  %v101 = extractvalue { i8*, i64 } %v12, 0
  %v210 = bitcast i8* %v101 to float*
  %v211 = getelementptr inbounds float, float* %v210, i64 %v79
  %v102 = bitcast float* %v211 to i8*
  %v212 = bitcast i8* %v102 to float*
  %v103 = load float, float* %v212, align 4
  %v104 = fadd contract float %v103, %v100
  %v213 = bitcast i8* %v102 to float*
  store float %v104, float* %v213, align 4
  br label %bb2
bb7:
  %v214 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v215 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v214, i32 0, i32 0
  %v105 = bitcast { i32, i32 }* %v215 to i8*
  %v216 = bitcast i8* %v105 to { i32, i32 }*
  %v217 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v216, i32 0, i32 0
  %v106 = bitcast i32* %v217 to i8*
  %v218 = bitcast i8* %v106 to i32*
  %v107 = load i32, i32* %v218, align 4
  %v219 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v220 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v219, i32 0, i32 0
  %v108 = bitcast { i32, i32 }* %v220 to i8*
  %v221 = bitcast i8* %v108 to { i32, i32 }*
  %v222 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v221, i32 0, i32 1
  %v109 = bitcast i32* %v222 to i8*
  %v223 = bitcast i8* %v109 to i32*
  %v110 = load i32, i32* %v223, align 4
  %v224 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v225 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v224, i32 0, i32 1
  %v111 = bitcast i64* %v225 to i8*
  %v226 = bitcast i8* %v111 to i64*
  %v112 = load i64, i64* %v226, align 8
  %v227 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v228 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v227, i32 0, i32 2
  %v113 = bitcast i1* %v228 to i8*
  %v229 = bitcast i8* %v113 to i1*
  %v114 = load i1, i1* %v229, align 1
  br label %bb2
bb8:
  %v115 = add i32 %v36, %v135
  %v116 = sub i32 %v37, 1
  %v117 = insertvalue { i32, i32 } undef, i32 1, 0
  %v118 = insertvalue { i32, i32 } %v117, i32 %v36, 1
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb10
bb9:
  %v121 = insertvalue { i32, i32 } undef, i32 0, 0
  %v122 = extractvalue { i32, i32 } %v121, 0
  %v123 = extractvalue { i32, i32 } %v121, 1
  br label %bb10
bb10:
  %v124 = phi i32 [ %v115, %bb8 ], [ %v36, %bb9 ]
  %v125 = phi i32 [ %v116, %bb8 ], [ %v37, %bb9 ]
  %v126 = phi i32 [ %v119, %bb8 ], [ %v122, %bb9 ]
  %v127 = phi i32 [ %v120, %bb8 ], [ %v123, %bb9 ]
  %v128 = insertvalue { i32, i32 } undef, i32 %v126, 0
  %v129 = insertvalue { i32, i32 } %v128, i32 %v127, 1
  %v130 = extractvalue { i32, i32 } %v129, 0
  %v131 = zext i32 %v130 to i64
  %v132 = icmp eq i64 %v131, 0
  br i1 %v132, label %bb5, label %bb11
bb11:
  %v133 = icmp eq i64 %v131, 1
  br i1 %v133, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v134 = trunc i64 %v51 to i32
  br label %bb14
bb14:
  %v135 = phi i32 [ 4294967295, %bb12 ], [ %v134, %bb13 ]
  %v136 = icmp ugt i32 %v37, 0
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb9, label %bb8
}

define void @gpu_copy_cuda_entry_9e8456c9ad6620bb(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v138 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v138 to i8*
  %v139 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v15 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v139 to i8*
  %v140 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v140 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v141 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v141, align 4
  br label %bb1
bb1:
  %v142 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v143 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v142, i32 0, i32 0
  %v18 = bitcast i32* %v143 to i8*
  %v144 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v144, align 4
  %v145 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v146 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v145, i32 0, i32 0
  %v20 = bitcast i32* %v146 to i8*
  %v147 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v147, align 4
  %v148 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v149 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v148, i32 0, i32 1
  %v22 = bitcast i32* %v149 to i8*
  %v150 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v150, align 4
  %v24 = mul i32 %v21, %v23
  %v151 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v152 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v151, i32 0, i32 2
  %v25 = bitcast i32* %v152 to i8*
  %v153 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v153, align 4
  %v27 = mul i32 %v24, %v26
  %v154 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v155 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v154, i32 0, i32 3
  %v28 = bitcast i32* %v155 to i8*
  %v156 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v156, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = insertvalue { i32, i32 } undef, i32 %v19, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v30, 1
  %v33 = extractvalue { i32, i32 } %v32, 0
  %v34 = extractvalue { i32, i32 } %v32, 1
  %v35 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v33, i32 %v34, i64 16776960) #0
  %v158 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v35, { { i32, i32 }, i64, i1, [7 x i8] }* %v158, align 8
  br label %bb7
bb2:
  %v36 = phi i32 [ %v122, %bb6 ], [ %v105, %bb7 ]
  %v37 = phi i32 [ %v123, %bb6 ], [ %v108, %bb7 ]
  %v38 = add i64 %v110, 1
  %v159 = alloca i64, align 8
  %v39 = bitcast i64* %v159 to i8*
  %v160 = bitcast i8* %v39 to i64*
  store i64 %v38, i64* %v160, align 8
  %v161 = bitcast i8* %v39 to { i64 }*
  %v40 = load { i64 }, { i64 }* %v161, align 8
  %v41 = extractvalue { i64 } %v40, 0
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  %v162 = alloca { i64 }, align 8
  %v47 = bitcast { i64 }* %v162 to i8*
  %v163 = bitcast i8* %v47 to { i64 }*
  store { i64 } %v40, { i64 }* %v163, align 8
  %v48 = getelementptr inbounds i8, i8* %v47, i64 0
  %v164 = bitcast i8* %v48 to { { i64 } }*
  %v49 = load { { i64 } }, { { i64 } }* %v164, align 8
  %v165 = alloca { { i64 } }, align 8
  %v50 = bitcast { { i64 } }* %v165 to i8*
  %v166 = bitcast i8* %v50 to { { i64 } }*
  store { { i64 } } %v49, { { i64 } }* %v166, align 8
  %v167 = bitcast i8* %v50 to i64*
  %v51 = load i64, i64* %v167, align 8
  %v52 = icmp ugt i64 %v51, 4294967295
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v54 = extractvalue { i32, i32 } %v127, 1
  %v55 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v54) #0
  %v168 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v55, { i32, i32, i32, i32 }* %v168, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v169 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v170 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v169, i32 0, i32 0
  %v56 = bitcast i32* %v170 to i8*
  %v171 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v171, align 4
  %v172 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v173 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v172, i32 0, i32 1
  %v58 = bitcast i32* %v173 to i8*
  %v174 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v174, align 4
  %v175 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v176 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v175, i32 0, i32 2
  %v60 = bitcast i32* %v176 to i8*
  %v177 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v177, align 4
  %v178 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v179 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v178, i32 0, i32 3
  %v62 = bitcast i32* %v179 to i8*
  %v180 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v180, align 4
  %v181 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v182 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v181, i32 0, i32 4
  %v64 = bitcast i32* %v182 to i8*
  %v183 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v183, align 4
  %v66 = mul i32 %v57, %v65
  %v184 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v185 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v184, i32 0, i32 5
  %v67 = bitcast i32* %v185 to i8*
  %v186 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v186, align 4
  %v69 = mul i32 %v59, %v68
  %v70 = add i32 %v66, %v69
  %v187 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v188 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v187, i32 0, i32 6
  %v71 = bitcast i32* %v188 to i8*
  %v189 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v189, align 4
  %v73 = mul i32 %v61, %v72
  %v74 = add i32 %v70, %v73
  %v190 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v191 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v190, i32 0, i32 7
  %v75 = bitcast i32* %v191 to i8*
  %v192 = bitcast i8* %v75 to i32*
  %v76 = load i32, i32* %v192, align 4
  %v77 = mul i32 %v63, %v76
  %v78 = add i32 %v74, %v77
  %v79 = zext i32 %v78 to i64
  %v193 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v194 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v193, i32 0, i32 4
  %v80 = bitcast i32* %v194 to i8*
  %v195 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v195, align 4
  %v82 = mul i32 %v57, %v81
  %v196 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v197 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v196, i32 0, i32 5
  %v83 = bitcast i32* %v197 to i8*
  %v198 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v198, align 4
  %v85 = mul i32 %v59, %v84
  %v86 = add i32 %v82, %v85
  %v199 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v200 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v199, i32 0, i32 6
  %v87 = bitcast i32* %v200 to i8*
  %v201 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v201, align 4
  %v89 = mul i32 %v61, %v88
  %v90 = add i32 %v86, %v89
  %v202 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v203 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v202, i32 0, i32 7
  %v91 = bitcast i32* %v203 to i8*
  %v204 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v204, align 4
  %v93 = mul i32 %v63, %v92
  %v94 = add i32 %v90, %v93
  %v95 = zext i32 %v94 to i64
  %v96 = extractvalue { i8*, i64 } %v13, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { i8*, i64 } %v13, 0
  %v205 = bitcast i8* %v98 to float*
  %v206 = getelementptr inbounds float, float* %v205, i64 %v95
  %v99 = bitcast float* %v206 to i8*
  %v207 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v207, align 4
  %v101 = extractvalue { i8*, i64 } %v12, 0
  %v208 = bitcast i8* %v101 to float*
  %v209 = getelementptr inbounds float, float* %v208, i64 %v79
  %v102 = bitcast float* %v209 to i8*
  %v210 = bitcast i8* %v102 to float*
  store float %v100, float* %v210, align 4
  br label %bb2
bb7:
  %v211 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v212 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v211, i32 0, i32 0
  %v103 = bitcast { i32, i32 }* %v212 to i8*
  %v213 = bitcast i8* %v103 to { i32, i32 }*
  %v214 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v213, i32 0, i32 0
  %v104 = bitcast i32* %v214 to i8*
  %v215 = bitcast i8* %v104 to i32*
  %v105 = load i32, i32* %v215, align 4
  %v216 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v217 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v216, i32 0, i32 0
  %v106 = bitcast { i32, i32 }* %v217 to i8*
  %v218 = bitcast i8* %v106 to { i32, i32 }*
  %v219 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v218, i32 0, i32 1
  %v107 = bitcast i32* %v219 to i8*
  %v220 = bitcast i8* %v107 to i32*
  %v108 = load i32, i32* %v220, align 4
  %v221 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v222 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v221, i32 0, i32 1
  %v109 = bitcast i64* %v222 to i8*
  %v223 = bitcast i8* %v109 to i64*
  %v110 = load i64, i64* %v223, align 8
  %v224 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v225 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v224, i32 0, i32 2
  %v111 = bitcast i1* %v225 to i8*
  %v226 = bitcast i8* %v111 to i1*
  %v112 = load i1, i1* %v226, align 1
  br label %bb2
bb8:
  %v113 = add i32 %v36, %v133
  %v114 = sub i32 %v37, 1
  %v115 = insertvalue { i32, i32 } undef, i32 1, 0
  %v116 = insertvalue { i32, i32 } %v115, i32 %v36, 1
  %v117 = extractvalue { i32, i32 } %v116, 0
  %v118 = extractvalue { i32, i32 } %v116, 1
  br label %bb10
bb9:
  %v119 = insertvalue { i32, i32 } undef, i32 0, 0
  %v120 = extractvalue { i32, i32 } %v119, 0
  %v121 = extractvalue { i32, i32 } %v119, 1
  br label %bb10
bb10:
  %v122 = phi i32 [ %v113, %bb8 ], [ %v36, %bb9 ]
  %v123 = phi i32 [ %v114, %bb8 ], [ %v37, %bb9 ]
  %v124 = phi i32 [ %v117, %bb8 ], [ %v120, %bb9 ]
  %v125 = phi i32 [ %v118, %bb8 ], [ %v121, %bb9 ]
  %v126 = insertvalue { i32, i32 } undef, i32 %v124, 0
  %v127 = insertvalue { i32, i32 } %v126, i32 %v125, 1
  %v128 = extractvalue { i32, i32 } %v127, 0
  %v129 = zext i32 %v128 to i64
  %v130 = icmp eq i64 %v129, 0
  br i1 %v130, label %bb5, label %bb11
bb11:
  %v131 = icmp eq i64 %v129, 1
  br i1 %v131, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v132 = trunc i64 %v51 to i32
  br label %bb14
bb14:
  %v133 = phi i32 [ 4294967295, %bb12 ], [ %v132, %bb13 ]
  %v134 = icmp ugt i32 %v37, 0
  %v135 = xor i1 %v134, 1
  br i1 %v135, label %bb9, label %bb8
}

define void @gpu_div_cuda_entry_b9c2b235c2039e1d(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v140 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v140 to i8*
  %v141 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v15 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v141 to i8*
  %v142 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v142 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v143 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v143, align 4
  br label %bb1
bb1:
  %v144 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v145 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v144, i32 0, i32 0
  %v18 = bitcast i32* %v145 to i8*
  %v146 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v146, align 4
  %v147 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 0
  %v20 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v149, align 4
  %v150 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v151 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v150, i32 0, i32 1
  %v22 = bitcast i32* %v151 to i8*
  %v152 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v152, align 4
  %v24 = mul i32 %v21, %v23
  %v153 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v154 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v153, i32 0, i32 2
  %v25 = bitcast i32* %v154 to i8*
  %v155 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v155, align 4
  %v27 = mul i32 %v24, %v26
  %v156 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v157 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v156, i32 0, i32 3
  %v28 = bitcast i32* %v157 to i8*
  %v158 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v158, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = insertvalue { i32, i32 } undef, i32 %v19, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v30, 1
  %v33 = extractvalue { i32, i32 } %v32, 0
  %v34 = extractvalue { i32, i32 } %v32, 1
  %v35 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v33, i32 %v34, i64 16776960) #0
  %v160 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v35, { { i32, i32 }, i64, i1, [7 x i8] }* %v160, align 8
  br label %bb7
bb2:
  %v36 = phi i32 [ %v124, %bb6 ], [ %v107, %bb7 ]
  %v37 = phi i32 [ %v125, %bb6 ], [ %v110, %bb7 ]
  %v38 = add i64 %v112, 1
  %v161 = alloca i64, align 8
  %v39 = bitcast i64* %v161 to i8*
  %v162 = bitcast i8* %v39 to i64*
  store i64 %v38, i64* %v162, align 8
  %v163 = bitcast i8* %v39 to { i64 }*
  %v40 = load { i64 }, { i64 }* %v163, align 8
  %v41 = extractvalue { i64 } %v40, 0
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  %v164 = alloca { i64 }, align 8
  %v47 = bitcast { i64 }* %v164 to i8*
  %v165 = bitcast i8* %v47 to { i64 }*
  store { i64 } %v40, { i64 }* %v165, align 8
  %v48 = getelementptr inbounds i8, i8* %v47, i64 0
  %v166 = bitcast i8* %v48 to { { i64 } }*
  %v49 = load { { i64 } }, { { i64 } }* %v166, align 8
  %v167 = alloca { { i64 } }, align 8
  %v50 = bitcast { { i64 } }* %v167 to i8*
  %v168 = bitcast i8* %v50 to { { i64 } }*
  store { { i64 } } %v49, { { i64 } }* %v168, align 8
  %v169 = bitcast i8* %v50 to i64*
  %v51 = load i64, i64* %v169, align 8
  %v52 = icmp ugt i64 %v51, 4294967295
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v54 = extractvalue { i32, i32 } %v129, 1
  %v55 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v54) #0
  %v170 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v55, { i32, i32, i32, i32 }* %v170, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v171 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v171, i32 0, i32 0
  %v56 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v173, align 4
  %v174 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v175 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v174, i32 0, i32 1
  %v58 = bitcast i32* %v175 to i8*
  %v176 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v176, align 4
  %v177 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v177, i32 0, i32 2
  %v60 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v181 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v180, i32 0, i32 3
  %v62 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v182, align 4
  %v183 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v183, i32 0, i32 4
  %v64 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v185, align 4
  %v66 = mul i32 %v57, %v65
  %v186 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v187 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v186, i32 0, i32 5
  %v67 = bitcast i32* %v187 to i8*
  %v188 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v188, align 4
  %v69 = mul i32 %v59, %v68
  %v70 = add i32 %v66, %v69
  %v189 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v190 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v189, i32 0, i32 6
  %v71 = bitcast i32* %v190 to i8*
  %v191 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v191, align 4
  %v73 = mul i32 %v61, %v72
  %v74 = add i32 %v70, %v73
  %v192 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v193 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v192, i32 0, i32 7
  %v75 = bitcast i32* %v193 to i8*
  %v194 = bitcast i8* %v75 to i32*
  %v76 = load i32, i32* %v194, align 4
  %v77 = mul i32 %v63, %v76
  %v78 = add i32 %v74, %v77
  %v79 = zext i32 %v78 to i64
  %v195 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v196 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v195, i32 0, i32 4
  %v80 = bitcast i32* %v196 to i8*
  %v197 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v197, align 4
  %v82 = mul i32 %v57, %v81
  %v198 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v199 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v198, i32 0, i32 5
  %v83 = bitcast i32* %v199 to i8*
  %v200 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v200, align 4
  %v85 = mul i32 %v59, %v84
  %v86 = add i32 %v82, %v85
  %v201 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v202 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v201, i32 0, i32 6
  %v87 = bitcast i32* %v202 to i8*
  %v203 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v203, align 4
  %v89 = mul i32 %v61, %v88
  %v90 = add i32 %v86, %v89
  %v204 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v205 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v204, i32 0, i32 7
  %v91 = bitcast i32* %v205 to i8*
  %v206 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v206, align 4
  %v93 = mul i32 %v63, %v92
  %v94 = add i32 %v90, %v93
  %v95 = zext i32 %v94 to i64
  %v96 = extractvalue { i8*, i64 } %v13, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { i8*, i64 } %v13, 0
  %v207 = bitcast i8* %v98 to float*
  %v208 = getelementptr inbounds float, float* %v207, i64 %v95
  %v99 = bitcast float* %v208 to i8*
  %v209 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v209, align 4
  %v101 = extractvalue { i8*, i64 } %v12, 0
  %v210 = bitcast i8* %v101 to float*
  %v211 = getelementptr inbounds float, float* %v210, i64 %v79
  %v102 = bitcast float* %v211 to i8*
  %v212 = bitcast i8* %v102 to float*
  %v103 = load float, float* %v212, align 4
  %v104 = fdiv contract float %v103, %v100
  %v213 = bitcast i8* %v102 to float*
  store float %v104, float* %v213, align 4
  br label %bb2
bb7:
  %v214 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v215 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v214, i32 0, i32 0
  %v105 = bitcast { i32, i32 }* %v215 to i8*
  %v216 = bitcast i8* %v105 to { i32, i32 }*
  %v217 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v216, i32 0, i32 0
  %v106 = bitcast i32* %v217 to i8*
  %v218 = bitcast i8* %v106 to i32*
  %v107 = load i32, i32* %v218, align 4
  %v219 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v220 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v219, i32 0, i32 0
  %v108 = bitcast { i32, i32 }* %v220 to i8*
  %v221 = bitcast i8* %v108 to { i32, i32 }*
  %v222 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v221, i32 0, i32 1
  %v109 = bitcast i32* %v222 to i8*
  %v223 = bitcast i8* %v109 to i32*
  %v110 = load i32, i32* %v223, align 4
  %v224 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v225 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v224, i32 0, i32 1
  %v111 = bitcast i64* %v225 to i8*
  %v226 = bitcast i8* %v111 to i64*
  %v112 = load i64, i64* %v226, align 8
  %v227 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v228 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v227, i32 0, i32 2
  %v113 = bitcast i1* %v228 to i8*
  %v229 = bitcast i8* %v113 to i1*
  %v114 = load i1, i1* %v229, align 1
  br label %bb2
bb8:
  %v115 = add i32 %v36, %v135
  %v116 = sub i32 %v37, 1
  %v117 = insertvalue { i32, i32 } undef, i32 1, 0
  %v118 = insertvalue { i32, i32 } %v117, i32 %v36, 1
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb10
bb9:
  %v121 = insertvalue { i32, i32 } undef, i32 0, 0
  %v122 = extractvalue { i32, i32 } %v121, 0
  %v123 = extractvalue { i32, i32 } %v121, 1
  br label %bb10
bb10:
  %v124 = phi i32 [ %v115, %bb8 ], [ %v36, %bb9 ]
  %v125 = phi i32 [ %v116, %bb8 ], [ %v37, %bb9 ]
  %v126 = phi i32 [ %v119, %bb8 ], [ %v122, %bb9 ]
  %v127 = phi i32 [ %v120, %bb8 ], [ %v123, %bb9 ]
  %v128 = insertvalue { i32, i32 } undef, i32 %v126, 0
  %v129 = insertvalue { i32, i32 } %v128, i32 %v127, 1
  %v130 = extractvalue { i32, i32 } %v129, 0
  %v131 = zext i32 %v130 to i64
  %v132 = icmp eq i64 %v131, 0
  br i1 %v132, label %bb5, label %bb11
bb11:
  %v133 = icmp eq i64 %v131, 1
  br i1 %v133, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v134 = trunc i64 %v51 to i32
  br label %bb14
bb14:
  %v135 = phi i32 [ 4294967295, %bb12 ], [ %v134, %bb13 ]
  %v136 = icmp ugt i32 %v37, 0
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb9, label %bb8
}

define void @gpu_mul_cuda_entry_8999a608a654ff30(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v140 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v140 to i8*
  %v141 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v15 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v141 to i8*
  %v142 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v142 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v143 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v143, align 4
  br label %bb1
bb1:
  %v144 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v145 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v144, i32 0, i32 0
  %v18 = bitcast i32* %v145 to i8*
  %v146 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v146, align 4
  %v147 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 0
  %v20 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v149, align 4
  %v150 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v151 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v150, i32 0, i32 1
  %v22 = bitcast i32* %v151 to i8*
  %v152 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v152, align 4
  %v24 = mul i32 %v21, %v23
  %v153 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v154 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v153, i32 0, i32 2
  %v25 = bitcast i32* %v154 to i8*
  %v155 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v155, align 4
  %v27 = mul i32 %v24, %v26
  %v156 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v157 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v156, i32 0, i32 3
  %v28 = bitcast i32* %v157 to i8*
  %v158 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v158, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = insertvalue { i32, i32 } undef, i32 %v19, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v30, 1
  %v33 = extractvalue { i32, i32 } %v32, 0
  %v34 = extractvalue { i32, i32 } %v32, 1
  %v35 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v33, i32 %v34, i64 16776960) #0
  %v160 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v35, { { i32, i32 }, i64, i1, [7 x i8] }* %v160, align 8
  br label %bb7
bb2:
  %v36 = phi i32 [ %v124, %bb6 ], [ %v107, %bb7 ]
  %v37 = phi i32 [ %v125, %bb6 ], [ %v110, %bb7 ]
  %v38 = add i64 %v112, 1
  %v161 = alloca i64, align 8
  %v39 = bitcast i64* %v161 to i8*
  %v162 = bitcast i8* %v39 to i64*
  store i64 %v38, i64* %v162, align 8
  %v163 = bitcast i8* %v39 to { i64 }*
  %v40 = load { i64 }, { i64 }* %v163, align 8
  %v41 = extractvalue { i64 } %v40, 0
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  %v164 = alloca { i64 }, align 8
  %v47 = bitcast { i64 }* %v164 to i8*
  %v165 = bitcast i8* %v47 to { i64 }*
  store { i64 } %v40, { i64 }* %v165, align 8
  %v48 = getelementptr inbounds i8, i8* %v47, i64 0
  %v166 = bitcast i8* %v48 to { { i64 } }*
  %v49 = load { { i64 } }, { { i64 } }* %v166, align 8
  %v167 = alloca { { i64 } }, align 8
  %v50 = bitcast { { i64 } }* %v167 to i8*
  %v168 = bitcast i8* %v50 to { { i64 } }*
  store { { i64 } } %v49, { { i64 } }* %v168, align 8
  %v169 = bitcast i8* %v50 to i64*
  %v51 = load i64, i64* %v169, align 8
  %v52 = icmp ugt i64 %v51, 4294967295
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v54 = extractvalue { i32, i32 } %v129, 1
  %v55 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v54) #0
  %v170 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v55, { i32, i32, i32, i32 }* %v170, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v171 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v171, i32 0, i32 0
  %v56 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v173, align 4
  %v174 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v175 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v174, i32 0, i32 1
  %v58 = bitcast i32* %v175 to i8*
  %v176 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v176, align 4
  %v177 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v177, i32 0, i32 2
  %v60 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v181 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v180, i32 0, i32 3
  %v62 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v182, align 4
  %v183 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v183, i32 0, i32 4
  %v64 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v185, align 4
  %v66 = mul i32 %v57, %v65
  %v186 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v187 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v186, i32 0, i32 5
  %v67 = bitcast i32* %v187 to i8*
  %v188 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v188, align 4
  %v69 = mul i32 %v59, %v68
  %v70 = add i32 %v66, %v69
  %v189 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v190 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v189, i32 0, i32 6
  %v71 = bitcast i32* %v190 to i8*
  %v191 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v191, align 4
  %v73 = mul i32 %v61, %v72
  %v74 = add i32 %v70, %v73
  %v192 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v193 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v192, i32 0, i32 7
  %v75 = bitcast i32* %v193 to i8*
  %v194 = bitcast i8* %v75 to i32*
  %v76 = load i32, i32* %v194, align 4
  %v77 = mul i32 %v63, %v76
  %v78 = add i32 %v74, %v77
  %v79 = zext i32 %v78 to i64
  %v195 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v196 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v195, i32 0, i32 4
  %v80 = bitcast i32* %v196 to i8*
  %v197 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v197, align 4
  %v82 = mul i32 %v57, %v81
  %v198 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v199 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v198, i32 0, i32 5
  %v83 = bitcast i32* %v199 to i8*
  %v200 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v200, align 4
  %v85 = mul i32 %v59, %v84
  %v86 = add i32 %v82, %v85
  %v201 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v202 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v201, i32 0, i32 6
  %v87 = bitcast i32* %v202 to i8*
  %v203 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v203, align 4
  %v89 = mul i32 %v61, %v88
  %v90 = add i32 %v86, %v89
  %v204 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v205 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v204, i32 0, i32 7
  %v91 = bitcast i32* %v205 to i8*
  %v206 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v206, align 4
  %v93 = mul i32 %v63, %v92
  %v94 = add i32 %v90, %v93
  %v95 = zext i32 %v94 to i64
  %v96 = extractvalue { i8*, i64 } %v13, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { i8*, i64 } %v13, 0
  %v207 = bitcast i8* %v98 to float*
  %v208 = getelementptr inbounds float, float* %v207, i64 %v95
  %v99 = bitcast float* %v208 to i8*
  %v209 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v209, align 4
  %v101 = extractvalue { i8*, i64 } %v12, 0
  %v210 = bitcast i8* %v101 to float*
  %v211 = getelementptr inbounds float, float* %v210, i64 %v79
  %v102 = bitcast float* %v211 to i8*
  %v212 = bitcast i8* %v102 to float*
  %v103 = load float, float* %v212, align 4
  %v104 = fmul contract float %v103, %v100
  %v213 = bitcast i8* %v102 to float*
  store float %v104, float* %v213, align 4
  br label %bb2
bb7:
  %v214 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v215 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v214, i32 0, i32 0
  %v105 = bitcast { i32, i32 }* %v215 to i8*
  %v216 = bitcast i8* %v105 to { i32, i32 }*
  %v217 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v216, i32 0, i32 0
  %v106 = bitcast i32* %v217 to i8*
  %v218 = bitcast i8* %v106 to i32*
  %v107 = load i32, i32* %v218, align 4
  %v219 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v220 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v219, i32 0, i32 0
  %v108 = bitcast { i32, i32 }* %v220 to i8*
  %v221 = bitcast i8* %v108 to { i32, i32 }*
  %v222 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v221, i32 0, i32 1
  %v109 = bitcast i32* %v222 to i8*
  %v223 = bitcast i8* %v109 to i32*
  %v110 = load i32, i32* %v223, align 4
  %v224 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v225 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v224, i32 0, i32 1
  %v111 = bitcast i64* %v225 to i8*
  %v226 = bitcast i8* %v111 to i64*
  %v112 = load i64, i64* %v226, align 8
  %v227 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v228 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v227, i32 0, i32 2
  %v113 = bitcast i1* %v228 to i8*
  %v229 = bitcast i8* %v113 to i1*
  %v114 = load i1, i1* %v229, align 1
  br label %bb2
bb8:
  %v115 = add i32 %v36, %v135
  %v116 = sub i32 %v37, 1
  %v117 = insertvalue { i32, i32 } undef, i32 1, 0
  %v118 = insertvalue { i32, i32 } %v117, i32 %v36, 1
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb10
bb9:
  %v121 = insertvalue { i32, i32 } undef, i32 0, 0
  %v122 = extractvalue { i32, i32 } %v121, 0
  %v123 = extractvalue { i32, i32 } %v121, 1
  br label %bb10
bb10:
  %v124 = phi i32 [ %v115, %bb8 ], [ %v36, %bb9 ]
  %v125 = phi i32 [ %v116, %bb8 ], [ %v37, %bb9 ]
  %v126 = phi i32 [ %v119, %bb8 ], [ %v122, %bb9 ]
  %v127 = phi i32 [ %v120, %bb8 ], [ %v123, %bb9 ]
  %v128 = insertvalue { i32, i32 } undef, i32 %v126, 0
  %v129 = insertvalue { i32, i32 } %v128, i32 %v127, 1
  %v130 = extractvalue { i32, i32 } %v129, 0
  %v131 = zext i32 %v130 to i64
  %v132 = icmp eq i64 %v131, 0
  br i1 %v132, label %bb5, label %bb11
bb11:
  %v133 = icmp eq i64 %v131, 1
  br i1 %v133, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v134 = trunc i64 %v51 to i32
  br label %bb14
bb14:
  %v135 = phi i32 [ 4294967295, %bb12 ], [ %v134, %bb13 ]
  %v136 = icmp ugt i32 %v37, 0
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb9, label %bb8
}

define void @gpu_sub_cuda_entry_d42e3f2b5cf54553(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v140 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v140 to i8*
  %v141 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v15 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v141 to i8*
  %v142 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v142 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v143 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v143, align 4
  br label %bb1
bb1:
  %v144 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v145 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v144, i32 0, i32 0
  %v18 = bitcast i32* %v145 to i8*
  %v146 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v146, align 4
  %v147 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v148 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v147, i32 0, i32 0
  %v20 = bitcast i32* %v148 to i8*
  %v149 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v149, align 4
  %v150 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v151 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v150, i32 0, i32 1
  %v22 = bitcast i32* %v151 to i8*
  %v152 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v152, align 4
  %v24 = mul i32 %v21, %v23
  %v153 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v154 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v153, i32 0, i32 2
  %v25 = bitcast i32* %v154 to i8*
  %v155 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v155, align 4
  %v27 = mul i32 %v24, %v26
  %v156 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v157 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v156, i32 0, i32 3
  %v28 = bitcast i32* %v157 to i8*
  %v158 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v158, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = insertvalue { i32, i32 } undef, i32 %v19, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v30, 1
  %v33 = extractvalue { i32, i32 } %v32, 0
  %v34 = extractvalue { i32, i32 } %v32, 1
  %v35 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v33, i32 %v34, i64 16776960) #0
  %v160 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v35, { { i32, i32 }, i64, i1, [7 x i8] }* %v160, align 8
  br label %bb7
bb2:
  %v36 = phi i32 [ %v124, %bb6 ], [ %v107, %bb7 ]
  %v37 = phi i32 [ %v125, %bb6 ], [ %v110, %bb7 ]
  %v38 = add i64 %v112, 1
  %v161 = alloca i64, align 8
  %v39 = bitcast i64* %v161 to i8*
  %v162 = bitcast i8* %v39 to i64*
  store i64 %v38, i64* %v162, align 8
  %v163 = bitcast i8* %v39 to { i64 }*
  %v40 = load { i64 }, { i64 }* %v163, align 8
  %v41 = extractvalue { i64 } %v40, 0
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  %v164 = alloca { i64 }, align 8
  %v47 = bitcast { i64 }* %v164 to i8*
  %v165 = bitcast i8* %v47 to { i64 }*
  store { i64 } %v40, { i64 }* %v165, align 8
  %v48 = getelementptr inbounds i8, i8* %v47, i64 0
  %v166 = bitcast i8* %v48 to { { i64 } }*
  %v49 = load { { i64 } }, { { i64 } }* %v166, align 8
  %v167 = alloca { { i64 } }, align 8
  %v50 = bitcast { { i64 } }* %v167 to i8*
  %v168 = bitcast i8* %v50 to { { i64 } }*
  store { { i64 } } %v49, { { i64 } }* %v168, align 8
  %v169 = bitcast i8* %v50 to i64*
  %v51 = load i64, i64* %v169, align 8
  %v52 = icmp ugt i64 %v51, 4294967295
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v54 = extractvalue { i32, i32 } %v129, 1
  %v55 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v54) #0
  %v170 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v55, { i32, i32, i32, i32 }* %v170, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v171 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v171, i32 0, i32 0
  %v56 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v173, align 4
  %v174 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v175 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v174, i32 0, i32 1
  %v58 = bitcast i32* %v175 to i8*
  %v176 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v176, align 4
  %v177 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v177, i32 0, i32 2
  %v60 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v181 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v180, i32 0, i32 3
  %v62 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v182, align 4
  %v183 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v183, i32 0, i32 4
  %v64 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v185, align 4
  %v66 = mul i32 %v57, %v65
  %v186 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v187 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v186, i32 0, i32 5
  %v67 = bitcast i32* %v187 to i8*
  %v188 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v188, align 4
  %v69 = mul i32 %v59, %v68
  %v70 = add i32 %v66, %v69
  %v189 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v190 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v189, i32 0, i32 6
  %v71 = bitcast i32* %v190 to i8*
  %v191 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v191, align 4
  %v73 = mul i32 %v61, %v72
  %v74 = add i32 %v70, %v73
  %v192 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v193 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v192, i32 0, i32 7
  %v75 = bitcast i32* %v193 to i8*
  %v194 = bitcast i8* %v75 to i32*
  %v76 = load i32, i32* %v194, align 4
  %v77 = mul i32 %v63, %v76
  %v78 = add i32 %v74, %v77
  %v79 = zext i32 %v78 to i64
  %v195 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v196 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v195, i32 0, i32 4
  %v80 = bitcast i32* %v196 to i8*
  %v197 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v197, align 4
  %v82 = mul i32 %v57, %v81
  %v198 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v199 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v198, i32 0, i32 5
  %v83 = bitcast i32* %v199 to i8*
  %v200 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v200, align 4
  %v85 = mul i32 %v59, %v84
  %v86 = add i32 %v82, %v85
  %v201 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v202 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v201, i32 0, i32 6
  %v87 = bitcast i32* %v202 to i8*
  %v203 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v203, align 4
  %v89 = mul i32 %v61, %v88
  %v90 = add i32 %v86, %v89
  %v204 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v205 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v204, i32 0, i32 7
  %v91 = bitcast i32* %v205 to i8*
  %v206 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v206, align 4
  %v93 = mul i32 %v63, %v92
  %v94 = add i32 %v90, %v93
  %v95 = zext i32 %v94 to i64
  %v96 = extractvalue { i8*, i64 } %v13, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { i8*, i64 } %v13, 0
  %v207 = bitcast i8* %v98 to float*
  %v208 = getelementptr inbounds float, float* %v207, i64 %v95
  %v99 = bitcast float* %v208 to i8*
  %v209 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v209, align 4
  %v101 = extractvalue { i8*, i64 } %v12, 0
  %v210 = bitcast i8* %v101 to float*
  %v211 = getelementptr inbounds float, float* %v210, i64 %v79
  %v102 = bitcast float* %v211 to i8*
  %v212 = bitcast i8* %v102 to float*
  %v103 = load float, float* %v212, align 4
  %v104 = fsub contract float %v103, %v100
  %v213 = bitcast i8* %v102 to float*
  store float %v104, float* %v213, align 4
  br label %bb2
bb7:
  %v214 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v215 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v214, i32 0, i32 0
  %v105 = bitcast { i32, i32 }* %v215 to i8*
  %v216 = bitcast i8* %v105 to { i32, i32 }*
  %v217 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v216, i32 0, i32 0
  %v106 = bitcast i32* %v217 to i8*
  %v218 = bitcast i8* %v106 to i32*
  %v107 = load i32, i32* %v218, align 4
  %v219 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v220 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v219, i32 0, i32 0
  %v108 = bitcast { i32, i32 }* %v220 to i8*
  %v221 = bitcast i8* %v108 to { i32, i32 }*
  %v222 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v221, i32 0, i32 1
  %v109 = bitcast i32* %v222 to i8*
  %v223 = bitcast i8* %v109 to i32*
  %v110 = load i32, i32* %v223, align 4
  %v224 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v225 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v224, i32 0, i32 1
  %v111 = bitcast i64* %v225 to i8*
  %v226 = bitcast i8* %v111 to i64*
  %v112 = load i64, i64* %v226, align 8
  %v227 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v228 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v227, i32 0, i32 2
  %v113 = bitcast i1* %v228 to i8*
  %v229 = bitcast i8* %v113 to i1*
  %v114 = load i1, i1* %v229, align 1
  br label %bb2
bb8:
  %v115 = add i32 %v36, %v135
  %v116 = sub i32 %v37, 1
  %v117 = insertvalue { i32, i32 } undef, i32 1, 0
  %v118 = insertvalue { i32, i32 } %v117, i32 %v36, 1
  %v119 = extractvalue { i32, i32 } %v118, 0
  %v120 = extractvalue { i32, i32 } %v118, 1
  br label %bb10
bb9:
  %v121 = insertvalue { i32, i32 } undef, i32 0, 0
  %v122 = extractvalue { i32, i32 } %v121, 0
  %v123 = extractvalue { i32, i32 } %v121, 1
  br label %bb10
bb10:
  %v124 = phi i32 [ %v115, %bb8 ], [ %v36, %bb9 ]
  %v125 = phi i32 [ %v116, %bb8 ], [ %v37, %bb9 ]
  %v126 = phi i32 [ %v119, %bb8 ], [ %v122, %bb9 ]
  %v127 = phi i32 [ %v120, %bb8 ], [ %v123, %bb9 ]
  %v128 = insertvalue { i32, i32 } undef, i32 %v126, 0
  %v129 = insertvalue { i32, i32 } %v128, i32 %v127, 1
  %v130 = extractvalue { i32, i32 } %v129, 0
  %v131 = zext i32 %v130 to i64
  %v132 = icmp eq i64 %v131, 0
  br i1 %v132, label %bb5, label %bb11
bb11:
  %v133 = icmp eq i64 %v131, 1
  br i1 %v133, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v134 = trunc i64 %v51 to i32
  br label %bb14
bb14:
  %v135 = phi i32 [ 4294967295, %bb12 ], [ %v134, %bb13 ]
  %v136 = icmp ugt i32 %v37, 0
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb9, label %bb8
}

define void @gpu_copy_with_offsets_cuda_entry_f31750be5f3ce13a(i8* %v0, i8* %v1, i8* %v2, i8* %v3, i64 %v4, i8* %v5, i64 %v6) #0 {
entry:
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  %v9 = insertvalue { i8*, i64 } undef, i8* %v5, 0
  %v10 = insertvalue { i8*, i64 } %v9, i64 %v6, 1
  br label %bb0
bb0:
  %v11 = phi i8* [ %v0, %entry ]
  %v12 = phi i8* [ %v1, %entry ]
  %v13 = phi i8* [ %v2, %entry ]
  %v14 = phi { i8*, i64 } [ %v8, %entry ]
  %v15 = phi { i8*, i64 } [ %v10, %entry ]
  %v146 = alloca { i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32 }* %v146 to i8*
  %v147 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v17 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v147 to i8*
  %v148 = alloca { i32, i32, i32, i32 }, align 4
  %v18 = bitcast { i32, i32, i32, i32 }* %v148 to i8*
  %v19 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v149 = bitcast i8* %v16 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v19, { i32, i32, i32 }* %v149, align 4
  br label %bb1
bb1:
  %v150 = bitcast i8* %v16 to { i32, i32, i32 }*
  %v151 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v150, i32 0, i32 0
  %v20 = bitcast i32* %v151 to i8*
  %v152 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v152, align 4
  %v153 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v154 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v153, i32 0, i32 0
  %v22 = bitcast i32* %v154 to i8*
  %v155 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v155, align 4
  %v156 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v157 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v156, i32 0, i32 1
  %v24 = bitcast i32* %v157 to i8*
  %v158 = bitcast i8* %v24 to i32*
  %v25 = load i32, i32* %v158, align 4
  %v26 = mul i32 %v23, %v25
  %v159 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v160 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v159, i32 0, i32 2
  %v27 = bitcast i32* %v160 to i8*
  %v161 = bitcast i8* %v27 to i32*
  %v28 = load i32, i32* %v161, align 4
  %v29 = mul i32 %v26, %v28
  %v162 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v163 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v162, i32 0, i32 3
  %v30 = bitcast i32* %v163 to i8*
  %v164 = bitcast i8* %v30 to i32*
  %v31 = load i32, i32* %v164, align 4
  %v32 = mul i32 %v29, %v31
  %v33 = insertvalue { i32, i32 } undef, i32 %v21, 0
  %v34 = insertvalue { i32, i32 } %v33, i32 %v32, 1
  %v35 = extractvalue { i32, i32 } %v34, 0
  %v36 = extractvalue { i32, i32 } %v34, 1
  %v37 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v35, i32 %v36, i64 16776960) #0
  %v166 = bitcast i8* %v17 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v37, { { i32, i32 }, i64, i1, [7 x i8] }* %v166, align 8
  br label %bb7
bb2:
  %v38 = phi i32 [ %v130, %bb6 ], [ %v113, %bb7 ]
  %v39 = phi i32 [ %v131, %bb6 ], [ %v116, %bb7 ]
  %v40 = add i64 %v118, 1
  %v167 = alloca i64, align 8
  %v41 = bitcast i64* %v167 to i8*
  %v168 = bitcast i8* %v41 to i64*
  store i64 %v40, i64* %v168, align 8
  %v169 = bitcast i8* %v41 to { i64 }*
  %v42 = load { i64 }, { i64 }* %v169, align 8
  %v43 = extractvalue { i64 } %v42, 0
  %v44 = sub i64 %v43, 0
  %v45 = icmp ule i64 %v44, 0
  %v46 = add i64 %v44, 0
  %v47 = select i1 %v45, i64 %v46, i64 1
  %v48 = icmp eq i64 %v47, 1
  %v170 = alloca { i64 }, align 8
  %v49 = bitcast { i64 }* %v170 to i8*
  %v171 = bitcast i8* %v49 to { i64 }*
  store { i64 } %v42, { i64 }* %v171, align 8
  %v50 = getelementptr inbounds i8, i8* %v49, i64 0
  %v172 = bitcast i8* %v50 to { { i64 } }*
  %v51 = load { { i64 } }, { { i64 } }* %v172, align 8
  %v173 = alloca { { i64 } }, align 8
  %v52 = bitcast { { i64 } }* %v173 to i8*
  %v174 = bitcast i8* %v52 to { { i64 } }*
  store { { i64 } } %v51, { { i64 } }* %v174, align 8
  %v175 = bitcast i8* %v52 to i64*
  %v53 = load i64, i64* %v175, align 8
  %v54 = icmp ugt i64 %v53, 4294967295
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v56 = extractvalue { i32, i32 } %v135, 1
  %v57 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v12, i32 %v56) #0
  %v176 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v57, { i32, i32, i32, i32 }* %v176, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v177 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v177, i32 0, i32 0
  %v58 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v181 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v180, i32 0, i32 1
  %v60 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v182, align 4
  %v183 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v183, i32 0, i32 2
  %v62 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v185, align 4
  %v186 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v187 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v186, i32 0, i32 3
  %v64 = bitcast i32* %v187 to i8*
  %v188 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v188, align 4
  %v189 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v190 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v189, i32 0, i32 4
  %v66 = bitcast i32* %v190 to i8*
  %v191 = bitcast i8* %v66 to i32*
  %v67 = load i32, i32* %v191, align 4
  %v68 = mul i32 %v59, %v67
  %v192 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v193 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v192, i32 0, i32 5
  %v69 = bitcast i32* %v193 to i8*
  %v194 = bitcast i8* %v69 to i32*
  %v70 = load i32, i32* %v194, align 4
  %v71 = mul i32 %v61, %v70
  %v72 = add i32 %v68, %v71
  %v195 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v196 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v195, i32 0, i32 6
  %v73 = bitcast i32* %v196 to i8*
  %v197 = bitcast i8* %v73 to i32*
  %v74 = load i32, i32* %v197, align 4
  %v75 = mul i32 %v63, %v74
  %v76 = add i32 %v72, %v75
  %v198 = bitcast i8* %v12 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v199 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v198, i32 0, i32 7
  %v77 = bitcast i32* %v199 to i8*
  %v200 = bitcast i8* %v77 to i32*
  %v78 = load i32, i32* %v200, align 4
  %v79 = mul i32 %v65, %v78
  %v80 = add i32 %v76, %v79
  %v201 = bitcast i8* %v13 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v202 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v201, i32 0, i32 4
  %v81 = bitcast i32* %v202 to i8*
  %v203 = bitcast i8* %v81 to i32*
  %v82 = load i32, i32* %v203, align 4
  %v83 = mul i32 %v59, %v82
  %v204 = bitcast i8* %v13 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v205 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v204, i32 0, i32 5
  %v84 = bitcast i32* %v205 to i8*
  %v206 = bitcast i8* %v84 to i32*
  %v85 = load i32, i32* %v206, align 4
  %v86 = mul i32 %v61, %v85
  %v87 = add i32 %v83, %v86
  %v207 = bitcast i8* %v13 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v208 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v207, i32 0, i32 6
  %v88 = bitcast i32* %v208 to i8*
  %v209 = bitcast i8* %v88 to i32*
  %v89 = load i32, i32* %v209, align 4
  %v90 = mul i32 %v63, %v89
  %v91 = add i32 %v87, %v90
  %v210 = bitcast i8* %v13 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v211 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v210, i32 0, i32 7
  %v92 = bitcast i32* %v211 to i8*
  %v212 = bitcast i8* %v92 to i32*
  %v93 = load i32, i32* %v212, align 4
  %v94 = mul i32 %v65, %v93
  %v95 = add i32 %v91, %v94
  %v213 = bitcast i8* %v11 to { i32, i32, i32, i32 }*
  %v214 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v213, i32 0, i32 0
  %v96 = bitcast i32* %v214 to i8*
  %v215 = bitcast i8* %v96 to i32*
  %v97 = load i32, i32* %v215, align 4
  %v98 = add i32 %v97, %v80
  %v99 = zext i32 %v98 to i64
  %v216 = bitcast i8* %v11 to { i32, i32, i32, i32 }*
  %v217 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v216, i32 0, i32 1
  %v100 = bitcast i32* %v217 to i8*
  %v218 = bitcast i8* %v100 to i32*
  %v101 = load i32, i32* %v218, align 4
  %v102 = add i32 %v101, %v95
  %v103 = zext i32 %v102 to i64
  %v104 = extractvalue { i8*, i64 } %v15, 1
  %v105 = icmp ult i64 %v103, %v104
  %v106 = extractvalue { i8*, i64 } %v15, 0
  %v219 = bitcast i8* %v106 to float*
  %v220 = getelementptr inbounds float, float* %v219, i64 %v103
  %v107 = bitcast float* %v220 to i8*
  %v221 = bitcast i8* %v107 to float*
  %v108 = load float, float* %v221, align 4
  %v109 = extractvalue { i8*, i64 } %v14, 0
  %v222 = bitcast i8* %v109 to float*
  %v223 = getelementptr inbounds float, float* %v222, i64 %v99
  %v110 = bitcast float* %v223 to i8*
  %v224 = bitcast i8* %v110 to float*
  store float %v108, float* %v224, align 4
  br label %bb2
bb7:
  %v225 = bitcast i8* %v17 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v226 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v225, i32 0, i32 0
  %v111 = bitcast { i32, i32 }* %v226 to i8*
  %v227 = bitcast i8* %v111 to { i32, i32 }*
  %v228 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v227, i32 0, i32 0
  %v112 = bitcast i32* %v228 to i8*
  %v229 = bitcast i8* %v112 to i32*
  %v113 = load i32, i32* %v229, align 4
  %v230 = bitcast i8* %v17 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v231 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v230, i32 0, i32 0
  %v114 = bitcast { i32, i32 }* %v231 to i8*
  %v232 = bitcast i8* %v114 to { i32, i32 }*
  %v233 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v232, i32 0, i32 1
  %v115 = bitcast i32* %v233 to i8*
  %v234 = bitcast i8* %v115 to i32*
  %v116 = load i32, i32* %v234, align 4
  %v235 = bitcast i8* %v17 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v236 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v235, i32 0, i32 1
  %v117 = bitcast i64* %v236 to i8*
  %v237 = bitcast i8* %v117 to i64*
  %v118 = load i64, i64* %v237, align 8
  %v238 = bitcast i8* %v17 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v239 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v238, i32 0, i32 2
  %v119 = bitcast i1* %v239 to i8*
  %v240 = bitcast i8* %v119 to i1*
  %v120 = load i1, i1* %v240, align 1
  br label %bb2
bb8:
  %v121 = add i32 %v38, %v141
  %v122 = sub i32 %v39, 1
  %v123 = insertvalue { i32, i32 } undef, i32 1, 0
  %v124 = insertvalue { i32, i32 } %v123, i32 %v38, 1
  %v125 = extractvalue { i32, i32 } %v124, 0
  %v126 = extractvalue { i32, i32 } %v124, 1
  br label %bb10
bb9:
  %v127 = insertvalue { i32, i32 } undef, i32 0, 0
  %v128 = extractvalue { i32, i32 } %v127, 0
  %v129 = extractvalue { i32, i32 } %v127, 1
  br label %bb10
bb10:
  %v130 = phi i32 [ %v121, %bb8 ], [ %v38, %bb9 ]
  %v131 = phi i32 [ %v122, %bb8 ], [ %v39, %bb9 ]
  %v132 = phi i32 [ %v125, %bb8 ], [ %v128, %bb9 ]
  %v133 = phi i32 [ %v126, %bb8 ], [ %v129, %bb9 ]
  %v134 = insertvalue { i32, i32 } undef, i32 %v132, 0
  %v135 = insertvalue { i32, i32 } %v134, i32 %v133, 1
  %v136 = extractvalue { i32, i32 } %v135, 0
  %v137 = zext i32 %v136 to i64
  %v138 = icmp eq i64 %v137, 0
  br i1 %v138, label %bb5, label %bb11
bb11:
  %v139 = icmp eq i64 %v137, 1
  br i1 %v139, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v140 = trunc i64 %v53 to i32
  br label %bb14
bb14:
  %v141 = phi i32 [ 4294967295, %bb12 ], [ %v140, %bb13 ]
  %v142 = icmp ugt i32 %v39, 0
  %v143 = xor i1 %v142, 1
  br i1 %v143, label %bb9, label %bb8
}

declare float @__nv_expf(float)

define void @gpu_tanh_cuda_entry_28094766a61b17b8(i8* %v0, i8* %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v4 = insertvalue { i8*, i64 } %v3, i64 %v2, 1
  br label %bb0
bb0:
  %v5 = phi i8* [ %v0, %entry ]
  %v6 = phi { i8*, i64 } [ %v4, %entry ]
  %v123 = alloca { i32, i32, i32 }, align 4
  %v7 = bitcast { i32, i32, i32 }* %v123 to i8*
  %v124 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v8 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v124 to i8*
  %v125 = alloca { i32, i32, i32, i32 }, align 4
  %v9 = bitcast { i32, i32, i32, i32 }* %v125 to i8*
  %v10 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v126 = bitcast i8* %v7 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v10, { i32, i32, i32 }* %v126, align 4
  br label %bb1
bb1:
  %v127 = bitcast i8* %v7 to { i32, i32, i32 }*
  %v128 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v127, i32 0, i32 0
  %v11 = bitcast i32* %v128 to i8*
  %v129 = bitcast i8* %v11 to i32*
  %v12 = load i32, i32* %v129, align 4
  %v130 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v131 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v130, i32 0, i32 0
  %v13 = bitcast i32* %v131 to i8*
  %v132 = bitcast i8* %v13 to i32*
  %v14 = load i32, i32* %v132, align 4
  %v133 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v134 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v133, i32 0, i32 1
  %v15 = bitcast i32* %v134 to i8*
  %v135 = bitcast i8* %v15 to i32*
  %v16 = load i32, i32* %v135, align 4
  %v17 = mul i32 %v14, %v16
  %v136 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v137 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v136, i32 0, i32 2
  %v18 = bitcast i32* %v137 to i8*
  %v138 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v138, align 4
  %v20 = mul i32 %v17, %v19
  %v139 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v140 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v139, i32 0, i32 3
  %v21 = bitcast i32* %v140 to i8*
  %v141 = bitcast i8* %v21 to i32*
  %v22 = load i32, i32* %v141, align 4
  %v23 = mul i32 %v20, %v22
  %v24 = insertvalue { i32, i32 } undef, i32 %v12, 0
  %v25 = insertvalue { i32, i32 } %v24, i32 %v23, 1
  %v26 = extractvalue { i32, i32 } %v25, 0
  %v27 = extractvalue { i32, i32 } %v25, 1
  %v28 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v26, i32 %v27, i64 16776960) #0
  %v143 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v28, { { i32, i32 }, i64, i1, [7 x i8] }* %v143, align 8
  br label %bb7
bb2:
  %v29 = phi i32 [ %v80, %bb7 ], [ %v97, %bb16 ]
  %v30 = phi i32 [ %v83, %bb7 ], [ %v98, %bb16 ]
  %v31 = add i64 %v85, 1
  %v144 = alloca i64, align 8
  %v32 = bitcast i64* %v144 to i8*
  %v145 = bitcast i8* %v32 to i64*
  store i64 %v31, i64* %v145, align 8
  %v146 = bitcast i8* %v32 to { i64 }*
  %v33 = load { i64 }, { i64 }* %v146, align 8
  %v34 = extractvalue { i64 } %v33, 0
  %v35 = sub i64 %v34, 0
  %v36 = icmp ule i64 %v35, 0
  %v37 = add i64 %v35, 0
  %v38 = select i1 %v36, i64 %v37, i64 1
  %v39 = icmp eq i64 %v38, 1
  %v147 = alloca { i64 }, align 8
  %v40 = bitcast { i64 }* %v147 to i8*
  %v148 = bitcast i8* %v40 to { i64 }*
  store { i64 } %v33, { i64 }* %v148, align 8
  %v41 = getelementptr inbounds i8, i8* %v40, i64 0
  %v149 = bitcast i8* %v41 to { { i64 } }*
  %v42 = load { { i64 } }, { { i64 } }* %v149, align 8
  %v150 = alloca { { i64 } }, align 8
  %v43 = bitcast { { i64 } }* %v150 to i8*
  %v151 = bitcast i8* %v43 to { { i64 } }*
  store { { i64 } } %v42, { { i64 } }* %v151, align 8
  %v152 = bitcast i8* %v43 to i64*
  %v44 = load i64, i64* %v152, align 8
  %v45 = icmp ugt i64 %v44, 4294967295
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v47 = extractvalue { i32, i32 } %v102, 1
  %v48 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v5, i32 %v47) #0
  %v153 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v48, { i32, i32, i32, i32 }* %v153, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v154 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v155 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v154, i32 0, i32 0
  %v49 = bitcast i32* %v155 to i8*
  %v156 = bitcast i8* %v49 to i32*
  %v50 = load i32, i32* %v156, align 4
  %v157 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v158 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v157, i32 0, i32 1
  %v51 = bitcast i32* %v158 to i8*
  %v159 = bitcast i8* %v51 to i32*
  %v52 = load i32, i32* %v159, align 4
  %v160 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v161 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v160, i32 0, i32 2
  %v53 = bitcast i32* %v161 to i8*
  %v162 = bitcast i8* %v53 to i32*
  %v54 = load i32, i32* %v162, align 4
  %v163 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v164 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v163, i32 0, i32 3
  %v55 = bitcast i32* %v164 to i8*
  %v165 = bitcast i8* %v55 to i32*
  %v56 = load i32, i32* %v165, align 4
  %v166 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v167 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v166, i32 0, i32 4
  %v57 = bitcast i32* %v167 to i8*
  %v168 = bitcast i8* %v57 to i32*
  %v58 = load i32, i32* %v168, align 4
  %v59 = mul i32 %v50, %v58
  %v169 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v170 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v169, i32 0, i32 5
  %v60 = bitcast i32* %v170 to i8*
  %v171 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v171, align 4
  %v62 = mul i32 %v52, %v61
  %v63 = add i32 %v59, %v62
  %v172 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v173 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v172, i32 0, i32 6
  %v64 = bitcast i32* %v173 to i8*
  %v174 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v174, align 4
  %v66 = mul i32 %v54, %v65
  %v67 = add i32 %v63, %v66
  %v175 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v176 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v175, i32 0, i32 7
  %v68 = bitcast i32* %v176 to i8*
  %v177 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v177, align 4
  %v70 = mul i32 %v56, %v69
  %v71 = add i32 %v67, %v70
  %v72 = zext i32 %v71 to i64
  %v73 = extractvalue { i8*, i64 } %v6, 0
  %v178 = bitcast i8* %v73 to float*
  %v179 = getelementptr inbounds float, float* %v178, i64 %v72
  %v74 = bitcast float* %v179 to i8*
  %v180 = bitcast i8* %v74 to float*
  %v75 = load float, float* %v180, align 4
  %v76 = fmul contract float -2.0, %v75
  %v77 = call float @__nv_expf(float %v76) #0
  br label %bb17
bb7:
  %v181 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v182 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v181, i32 0, i32 0
  %v78 = bitcast { i32, i32 }* %v182 to i8*
  %v183 = bitcast i8* %v78 to { i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v183, i32 0, i32 0
  %v79 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v185, align 4
  %v186 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v187 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v186, i32 0, i32 0
  %v81 = bitcast { i32, i32 }* %v187 to i8*
  %v188 = bitcast i8* %v81 to { i32, i32 }*
  %v189 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v188, i32 0, i32 1
  %v82 = bitcast i32* %v189 to i8*
  %v190 = bitcast i8* %v82 to i32*
  %v83 = load i32, i32* %v190, align 4
  %v191 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v192 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v191, i32 0, i32 1
  %v84 = bitcast i64* %v192 to i8*
  %v193 = bitcast i8* %v84 to i64*
  %v85 = load i64, i64* %v193, align 8
  %v194 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v195 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v194, i32 0, i32 2
  %v86 = bitcast i1* %v195 to i8*
  %v196 = bitcast i8* %v86 to i1*
  %v87 = load i1, i1* %v196, align 1
  br label %bb2
bb8:
  %v88 = add i32 %v29, %v108
  %v89 = sub i32 %v30, 1
  %v90 = insertvalue { i32, i32 } undef, i32 1, 0
  %v91 = insertvalue { i32, i32 } %v90, i32 %v29, 1
  %v92 = extractvalue { i32, i32 } %v91, 0
  %v93 = extractvalue { i32, i32 } %v91, 1
  br label %bb10
bb9:
  %v94 = insertvalue { i32, i32 } undef, i32 0, 0
  %v95 = extractvalue { i32, i32 } %v94, 0
  %v96 = extractvalue { i32, i32 } %v94, 1
  br label %bb10
bb10:
  %v97 = phi i32 [ %v88, %bb8 ], [ %v29, %bb9 ]
  %v98 = phi i32 [ %v89, %bb8 ], [ %v30, %bb9 ]
  %v99 = phi i32 [ %v92, %bb8 ], [ %v95, %bb9 ]
  %v100 = phi i32 [ %v93, %bb8 ], [ %v96, %bb9 ]
  %v101 = insertvalue { i32, i32 } undef, i32 %v99, 0
  %v102 = insertvalue { i32, i32 } %v101, i32 %v100, 1
  %v103 = extractvalue { i32, i32 } %v102, 0
  %v104 = zext i32 %v103 to i64
  %v105 = icmp eq i64 %v104, 0
  br i1 %v105, label %bb5, label %bb11
bb11:
  %v106 = icmp eq i64 %v104, 1
  br i1 %v106, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v107 = trunc i64 %v44 to i32
  br label %bb14
bb14:
  %v108 = phi i32 [ 4294967295, %bb12 ], [ %v107, %bb13 ]
  %v109 = icmp ugt i32 %v30, 0
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb9, label %bb8
bb15:
  br label %bb16
bb16:
  %v111 = phi float [ %v119, %bb15 ], [ %v116, %bb18 ]
  %v200 = bitcast i8* %v74 to float*
  store float %v111, float* %v200, align 4
  br label %bb2
bb17:
  %v112 = fmul contract float 2.0, %v75
  %v113 = call float @__nv_expf(float %v112) #0
  br label %bb18
bb18:
  %v114 = fsub contract float 1.0, %v77
  %v115 = fadd contract float 1.0, %v77
  %v116 = fdiv contract float %v114, %v115
  %v117 = fsub contract float %v113, 1.0
  %v118 = fadd contract float %v113, 1.0
  %v119 = fdiv contract float %v117, %v118
  %v120 = fcmp oge float %v75, 0.0
  %v121 = xor i1 %v120, 1
  br i1 %v121, label %bb15, label %bb16
}

define void @gpu_elu_cuda_entry_38a39043a6eb4b5c(i8* %v0, i8* %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v4 = insertvalue { i8*, i64 } %v3, i64 %v2, 1
  br label %bb0
bb0:
  %v5 = phi i8* [ %v0, %entry ]
  %v6 = phi { i8*, i64 } [ %v4, %entry ]
  %v115 = alloca { i32, i32, i32 }, align 4
  %v7 = bitcast { i32, i32, i32 }* %v115 to i8*
  %v116 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v8 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v116 to i8*
  %v117 = alloca { i32, i32, i32, i32 }, align 4
  %v9 = bitcast { i32, i32, i32, i32 }* %v117 to i8*
  %v10 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v118 = bitcast i8* %v7 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v10, { i32, i32, i32 }* %v118, align 4
  br label %bb1
bb1:
  %v119 = bitcast i8* %v7 to { i32, i32, i32 }*
  %v120 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v119, i32 0, i32 0
  %v11 = bitcast i32* %v120 to i8*
  %v121 = bitcast i8* %v11 to i32*
  %v12 = load i32, i32* %v121, align 4
  %v122 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v123 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v122, i32 0, i32 0
  %v13 = bitcast i32* %v123 to i8*
  %v124 = bitcast i8* %v13 to i32*
  %v14 = load i32, i32* %v124, align 4
  %v125 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v126 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v125, i32 0, i32 1
  %v15 = bitcast i32* %v126 to i8*
  %v127 = bitcast i8* %v15 to i32*
  %v16 = load i32, i32* %v127, align 4
  %v17 = mul i32 %v14, %v16
  %v128 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v129 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v128, i32 0, i32 2
  %v18 = bitcast i32* %v129 to i8*
  %v130 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v130, align 4
  %v20 = mul i32 %v17, %v19
  %v131 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v132 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v131, i32 0, i32 3
  %v21 = bitcast i32* %v132 to i8*
  %v133 = bitcast i8* %v21 to i32*
  %v22 = load i32, i32* %v133, align 4
  %v23 = mul i32 %v20, %v22
  %v24 = insertvalue { i32, i32 } undef, i32 %v12, 0
  %v25 = insertvalue { i32, i32 } %v24, i32 %v23, 1
  %v26 = extractvalue { i32, i32 } %v25, 0
  %v27 = extractvalue { i32, i32 } %v25, 1
  %v28 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v26, i32 %v27, i64 16776960) #0
  %v135 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v28, { { i32, i32 }, i64, i1, [7 x i8] }* %v135, align 8
  br label %bb10
bb2:
  %v29 = phi i32 [ %v99, %bb9 ], [ %v82, %bb10 ]
  %v30 = phi i32 [ %v100, %bb9 ], [ %v85, %bb10 ]
  %v31 = add i64 %v87, 1
  %v136 = alloca i64, align 8
  %v32 = bitcast i64* %v136 to i8*
  %v137 = bitcast i8* %v32 to i64*
  store i64 %v31, i64* %v137, align 8
  %v138 = bitcast i8* %v32 to { i64 }*
  %v33 = load { i64 }, { i64 }* %v138, align 8
  %v34 = extractvalue { i64 } %v33, 0
  %v35 = sub i64 %v34, 0
  %v36 = icmp ule i64 %v35, 0
  %v37 = add i64 %v35, 0
  %v38 = select i1 %v36, i64 %v37, i64 1
  %v39 = icmp eq i64 %v38, 1
  %v139 = alloca { i64 }, align 8
  %v40 = bitcast { i64 }* %v139 to i8*
  %v140 = bitcast i8* %v40 to { i64 }*
  store { i64 } %v33, { i64 }* %v140, align 8
  %v41 = getelementptr inbounds i8, i8* %v40, i64 0
  %v141 = bitcast i8* %v41 to { { i64 } }*
  %v42 = load { { i64 } }, { { i64 } }* %v141, align 8
  %v142 = alloca { { i64 } }, align 8
  %v43 = bitcast { { i64 } }* %v142 to i8*
  %v143 = bitcast i8* %v43 to { { i64 } }*
  store { { i64 } } %v42, { { i64 } }* %v143, align 8
  %v144 = bitcast i8* %v43 to i64*
  %v44 = load i64, i64* %v144, align 8
  %v45 = icmp ugt i64 %v44, 4294967295
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb16, label %bb15
bb3:
  unreachable
bb4:
  %v47 = extractvalue { i32, i32 } %v104, 1
  %v48 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v5, i32 %v47) #0
  %v145 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v48, { i32, i32, i32, i32 }* %v145, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v146 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v146, i32 0, i32 0
  %v49 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v49 to i32*
  %v50 = load i32, i32* %v148, align 4
  %v149 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v150 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v149, i32 0, i32 1
  %v51 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v51 to i32*
  %v52 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v153 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v152, i32 0, i32 2
  %v53 = bitcast i32* %v153 to i8*
  %v154 = bitcast i8* %v53 to i32*
  %v54 = load i32, i32* %v154, align 4
  %v155 = bitcast i8* %v9 to { i32, i32, i32, i32 }*
  %v156 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v155, i32 0, i32 3
  %v55 = bitcast i32* %v156 to i8*
  %v157 = bitcast i8* %v55 to i32*
  %v56 = load i32, i32* %v157, align 4
  %v158 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v159 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v158, i32 0, i32 4
  %v57 = bitcast i32* %v159 to i8*
  %v160 = bitcast i8* %v57 to i32*
  %v58 = load i32, i32* %v160, align 4
  %v59 = mul i32 %v50, %v58
  %v161 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v162 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v161, i32 0, i32 5
  %v60 = bitcast i32* %v162 to i8*
  %v163 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v163, align 4
  %v62 = mul i32 %v52, %v61
  %v63 = add i32 %v59, %v62
  %v164 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v165 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v164, i32 0, i32 6
  %v64 = bitcast i32* %v165 to i8*
  %v166 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v166, align 4
  %v66 = mul i32 %v54, %v65
  %v67 = add i32 %v63, %v66
  %v167 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v168 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v167, i32 0, i32 7
  %v68 = bitcast i32* %v168 to i8*
  %v169 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v169, align 4
  %v70 = mul i32 %v56, %v69
  %v71 = add i32 %v67, %v70
  %v72 = zext i32 %v71 to i64
  %v73 = extractvalue { i8*, i64 } %v6, 0
  %v170 = bitcast i8* %v73 to float*
  %v171 = getelementptr inbounds float, float* %v170, i64 %v72
  %v74 = bitcast float* %v171 to i8*
  %v172 = bitcast i8* %v74 to float*
  %v75 = load float, float* %v172, align 4
  %v76 = fcmp ogt float %v75, 0.0
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb8, label %bb7
bb7:
  br label %bb9
bb8:
  %v78 = call float @__nv_expf(float %v75) #0
  br label %bb18
bb9:
  %v79 = phi float [ %v75, %bb7 ], [ %v113, %bb18 ]
  %v173 = bitcast i8* %v74 to float*
  store float %v79, float* %v173, align 4
  br label %bb2
bb10:
  %v174 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v175 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, i32 0, i32 0
  %v80 = bitcast { i32, i32 }* %v175 to i8*
  %v176 = bitcast i8* %v80 to { i32, i32 }*
  %v177 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v176, i32 0, i32 0
  %v81 = bitcast i32* %v177 to i8*
  %v178 = bitcast i8* %v81 to i32*
  %v82 = load i32, i32* %v178, align 4
  %v179 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v180 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v179, i32 0, i32 0
  %v83 = bitcast { i32, i32 }* %v180 to i8*
  %v181 = bitcast i8* %v83 to { i32, i32 }*
  %v182 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v181, i32 0, i32 1
  %v84 = bitcast i32* %v182 to i8*
  %v183 = bitcast i8* %v84 to i32*
  %v85 = load i32, i32* %v183, align 4
  %v184 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v185 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v184, i32 0, i32 1
  %v86 = bitcast i64* %v185 to i8*
  %v186 = bitcast i8* %v86 to i64*
  %v87 = load i64, i64* %v186, align 8
  %v187 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v188 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v187, i32 0, i32 2
  %v88 = bitcast i1* %v188 to i8*
  %v189 = bitcast i8* %v88 to i1*
  %v89 = load i1, i1* %v189, align 1
  br label %bb2
bb11:
  %v90 = add i32 %v29, %v110
  %v91 = sub i32 %v30, 1
  %v92 = insertvalue { i32, i32 } undef, i32 1, 0
  %v93 = insertvalue { i32, i32 } %v92, i32 %v29, 1
  %v94 = extractvalue { i32, i32 } %v93, 0
  %v95 = extractvalue { i32, i32 } %v93, 1
  br label %bb13
bb12:
  %v96 = insertvalue { i32, i32 } undef, i32 0, 0
  %v97 = extractvalue { i32, i32 } %v96, 0
  %v98 = extractvalue { i32, i32 } %v96, 1
  br label %bb13
bb13:
  %v99 = phi i32 [ %v90, %bb11 ], [ %v29, %bb12 ]
  %v100 = phi i32 [ %v91, %bb11 ], [ %v30, %bb12 ]
  %v101 = phi i32 [ %v94, %bb11 ], [ %v97, %bb12 ]
  %v102 = phi i32 [ %v95, %bb11 ], [ %v98, %bb12 ]
  %v103 = insertvalue { i32, i32 } undef, i32 %v101, 0
  %v104 = insertvalue { i32, i32 } %v103, i32 %v102, 1
  %v105 = extractvalue { i32, i32 } %v104, 0
  %v106 = zext i32 %v105 to i64
  %v107 = icmp eq i64 %v106, 0
  br i1 %v107, label %bb5, label %bb14
bb14:
  %v108 = icmp eq i64 %v106, 1
  br i1 %v108, label %bb4, label %bb3
bb15:
  br label %bb17
bb16:
  %v109 = trunc i64 %v44 to i32
  br label %bb17
bb17:
  %v110 = phi i32 [ 4294967295, %bb15 ], [ %v109, %bb16 ]
  %v111 = icmp ugt i32 %v30, 0
  %v112 = xor i1 %v111, 1
  br i1 %v112, label %bb12, label %bb11
bb18:
  %v113 = fsub contract float %v78, 1.0
  br label %bb9
}

define void @gpu_elu_backward_cuda_entry_251ddb1b5f024100(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v144 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v144 to i8*
  %v145 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v15 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v145 to i8*
  %v146 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v146 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v147 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v147, align 4
  br label %bb1
bb1:
  %v148 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v149 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v148, i32 0, i32 0
  %v18 = bitcast i32* %v149 to i8*
  %v150 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v150, align 4
  %v151 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v152 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v151, i32 0, i32 0
  %v20 = bitcast i32* %v152 to i8*
  %v153 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v153, align 4
  %v154 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v155 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v154, i32 0, i32 1
  %v22 = bitcast i32* %v155 to i8*
  %v156 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v156, align 4
  %v24 = mul i32 %v21, %v23
  %v157 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v158 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v157, i32 0, i32 2
  %v25 = bitcast i32* %v158 to i8*
  %v159 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v159, align 4
  %v27 = mul i32 %v24, %v26
  %v160 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v161 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v160, i32 0, i32 3
  %v28 = bitcast i32* %v161 to i8*
  %v162 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v162, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = insertvalue { i32, i32 } undef, i32 %v19, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v30, 1
  %v33 = extractvalue { i32, i32 } %v32, 0
  %v34 = extractvalue { i32, i32 } %v32, 1
  %v35 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v33, i32 %v34, i64 16776960) #0
  %v164 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v35, { { i32, i32 }, i64, i1, [7 x i8] }* %v164, align 8
  br label %bb10
bb2:
  %v36 = phi i32 [ %v128, %bb9 ], [ %v111, %bb10 ]
  %v37 = phi i32 [ %v129, %bb9 ], [ %v114, %bb10 ]
  %v38 = add i64 %v116, 1
  %v165 = alloca i64, align 8
  %v39 = bitcast i64* %v165 to i8*
  %v166 = bitcast i8* %v39 to i64*
  store i64 %v38, i64* %v166, align 8
  %v167 = bitcast i8* %v39 to { i64 }*
  %v40 = load { i64 }, { i64 }* %v167, align 8
  %v41 = extractvalue { i64 } %v40, 0
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  %v168 = alloca { i64 }, align 8
  %v47 = bitcast { i64 }* %v168 to i8*
  %v169 = bitcast i8* %v47 to { i64 }*
  store { i64 } %v40, { i64 }* %v169, align 8
  %v48 = getelementptr inbounds i8, i8* %v47, i64 0
  %v170 = bitcast i8* %v48 to { { i64 } }*
  %v49 = load { { i64 } }, { { i64 } }* %v170, align 8
  %v171 = alloca { { i64 } }, align 8
  %v50 = bitcast { { i64 } }* %v171 to i8*
  %v172 = bitcast i8* %v50 to { { i64 } }*
  store { { i64 } } %v49, { { i64 } }* %v172, align 8
  %v173 = bitcast i8* %v50 to i64*
  %v51 = load i64, i64* %v173, align 8
  %v52 = icmp ugt i64 %v51, 4294967295
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb16, label %bb15
bb3:
  unreachable
bb4:
  %v54 = extractvalue { i32, i32 } %v133, 1
  %v55 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v54) #0
  %v174 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v55, { i32, i32, i32, i32 }* %v174, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v175 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v176 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v175, i32 0, i32 0
  %v56 = bitcast i32* %v176 to i8*
  %v177 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v177, align 4
  %v178 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v179 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v178, i32 0, i32 1
  %v58 = bitcast i32* %v179 to i8*
  %v180 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v180, align 4
  %v181 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v182 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v181, i32 0, i32 2
  %v60 = bitcast i32* %v182 to i8*
  %v183 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v183, align 4
  %v184 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v185 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v184, i32 0, i32 3
  %v62 = bitcast i32* %v185 to i8*
  %v186 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v186, align 4
  %v187 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v188 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v187, i32 0, i32 4
  %v64 = bitcast i32* %v188 to i8*
  %v189 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v189, align 4
  %v66 = mul i32 %v57, %v65
  %v190 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v191 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v190, i32 0, i32 5
  %v67 = bitcast i32* %v191 to i8*
  %v192 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v192, align 4
  %v69 = mul i32 %v59, %v68
  %v70 = add i32 %v66, %v69
  %v193 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v194 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v193, i32 0, i32 6
  %v71 = bitcast i32* %v194 to i8*
  %v195 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v195, align 4
  %v73 = mul i32 %v61, %v72
  %v74 = add i32 %v70, %v73
  %v196 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v197 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v196, i32 0, i32 7
  %v75 = bitcast i32* %v197 to i8*
  %v198 = bitcast i8* %v75 to i32*
  %v76 = load i32, i32* %v198, align 4
  %v77 = mul i32 %v63, %v76
  %v78 = add i32 %v74, %v77
  %v79 = zext i32 %v78 to i64
  %v199 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v200 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v199, i32 0, i32 4
  %v80 = bitcast i32* %v200 to i8*
  %v201 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v201, align 4
  %v82 = mul i32 %v57, %v81
  %v202 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v203 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v202, i32 0, i32 5
  %v83 = bitcast i32* %v203 to i8*
  %v204 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v204, align 4
  %v85 = mul i32 %v59, %v84
  %v86 = add i32 %v82, %v85
  %v205 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v206 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v205, i32 0, i32 6
  %v87 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v207, align 4
  %v89 = mul i32 %v61, %v88
  %v90 = add i32 %v86, %v89
  %v208 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v209 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v208, i32 0, i32 7
  %v91 = bitcast i32* %v209 to i8*
  %v210 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v210, align 4
  %v93 = mul i32 %v63, %v92
  %v94 = add i32 %v90, %v93
  %v95 = zext i32 %v94 to i64
  %v96 = extractvalue { i8*, i64 } %v13, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { i8*, i64 } %v13, 0
  %v211 = bitcast i8* %v98 to float*
  %v212 = getelementptr inbounds float, float* %v211, i64 %v95
  %v99 = bitcast float* %v212 to i8*
  %v213 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v213, align 4
  %v101 = fcmp ogt float %v100, 0.0
  %v102 = xor i1 %v101, 1
  br i1 %v102, label %bb8, label %bb7
bb7:
  br label %bb9
bb8:
  %v103 = fadd contract float %v100, 1.0
  br label %bb9
bb9:
  %v104 = phi float [ 1.0, %bb7 ], [ %v103, %bb8 ]
  %v105 = extractvalue { i8*, i64 } %v12, 0
  %v214 = bitcast i8* %v105 to float*
  %v215 = getelementptr inbounds float, float* %v214, i64 %v79
  %v106 = bitcast float* %v215 to i8*
  %v216 = bitcast i8* %v106 to float*
  %v107 = load float, float* %v216, align 4
  %v108 = fmul contract float %v107, %v104
  %v217 = bitcast i8* %v106 to float*
  store float %v108, float* %v217, align 4
  br label %bb2
bb10:
  %v218 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v219 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v218, i32 0, i32 0
  %v109 = bitcast { i32, i32 }* %v219 to i8*
  %v220 = bitcast i8* %v109 to { i32, i32 }*
  %v221 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v220, i32 0, i32 0
  %v110 = bitcast i32* %v221 to i8*
  %v222 = bitcast i8* %v110 to i32*
  %v111 = load i32, i32* %v222, align 4
  %v223 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v224 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v223, i32 0, i32 0
  %v112 = bitcast { i32, i32 }* %v224 to i8*
  %v225 = bitcast i8* %v112 to { i32, i32 }*
  %v226 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v225, i32 0, i32 1
  %v113 = bitcast i32* %v226 to i8*
  %v227 = bitcast i8* %v113 to i32*
  %v114 = load i32, i32* %v227, align 4
  %v228 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v229 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v228, i32 0, i32 1
  %v115 = bitcast i64* %v229 to i8*
  %v230 = bitcast i8* %v115 to i64*
  %v116 = load i64, i64* %v230, align 8
  %v231 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v232 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v231, i32 0, i32 2
  %v117 = bitcast i1* %v232 to i8*
  %v233 = bitcast i8* %v117 to i1*
  %v118 = load i1, i1* %v233, align 1
  br label %bb2
bb11:
  %v119 = add i32 %v36, %v139
  %v120 = sub i32 %v37, 1
  %v121 = insertvalue { i32, i32 } undef, i32 1, 0
  %v122 = insertvalue { i32, i32 } %v121, i32 %v36, 1
  %v123 = extractvalue { i32, i32 } %v122, 0
  %v124 = extractvalue { i32, i32 } %v122, 1
  br label %bb13
bb12:
  %v125 = insertvalue { i32, i32 } undef, i32 0, 0
  %v126 = extractvalue { i32, i32 } %v125, 0
  %v127 = extractvalue { i32, i32 } %v125, 1
  br label %bb13
bb13:
  %v128 = phi i32 [ %v119, %bb11 ], [ %v36, %bb12 ]
  %v129 = phi i32 [ %v120, %bb11 ], [ %v37, %bb12 ]
  %v130 = phi i32 [ %v123, %bb11 ], [ %v126, %bb12 ]
  %v131 = phi i32 [ %v124, %bb11 ], [ %v127, %bb12 ]
  %v132 = insertvalue { i32, i32 } undef, i32 %v130, 0
  %v133 = insertvalue { i32, i32 } %v132, i32 %v131, 1
  %v134 = extractvalue { i32, i32 } %v133, 0
  %v135 = zext i32 %v134 to i64
  %v136 = icmp eq i64 %v135, 0
  br i1 %v136, label %bb5, label %bb14
bb14:
  %v137 = icmp eq i64 %v135, 1
  br i1 %v137, label %bb4, label %bb3
bb15:
  br label %bb17
bb16:
  %v138 = trunc i64 %v51 to i32
  br label %bb17
bb17:
  %v139 = phi i32 [ 4294967295, %bb15 ], [ %v138, %bb16 ]
  %v140 = icmp ugt i32 %v37, 0
  %v141 = xor i1 %v140, 1
  br i1 %v141, label %bb12, label %bb11
}

define void @gpu_tanh_backward_cuda_entry_e1a55edc2786a0fd(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v142 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v142 to i8*
  %v143 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v15 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v143 to i8*
  %v144 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v144 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v145 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v145, align 4
  br label %bb1
bb1:
  %v146 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v147 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v146, i32 0, i32 0
  %v18 = bitcast i32* %v147 to i8*
  %v148 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v148, align 4
  %v149 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v150 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v149, i32 0, i32 0
  %v20 = bitcast i32* %v150 to i8*
  %v151 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v151, align 4
  %v152 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v153 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v152, i32 0, i32 1
  %v22 = bitcast i32* %v153 to i8*
  %v154 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v154, align 4
  %v24 = mul i32 %v21, %v23
  %v155 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v156 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v155, i32 0, i32 2
  %v25 = bitcast i32* %v156 to i8*
  %v157 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v157, align 4
  %v27 = mul i32 %v24, %v26
  %v158 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v159 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v158, i32 0, i32 3
  %v28 = bitcast i32* %v159 to i8*
  %v160 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v160, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = insertvalue { i32, i32 } undef, i32 %v19, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v30, 1
  %v33 = extractvalue { i32, i32 } %v32, 0
  %v34 = extractvalue { i32, i32 } %v32, 1
  %v35 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v33, i32 %v34, i64 16776960) #0
  %v162 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v35, { { i32, i32 }, i64, i1, [7 x i8] }* %v162, align 8
  br label %bb7
bb2:
  %v36 = phi i32 [ %v126, %bb6 ], [ %v109, %bb7 ]
  %v37 = phi i32 [ %v127, %bb6 ], [ %v112, %bb7 ]
  %v38 = add i64 %v114, 1
  %v163 = alloca i64, align 8
  %v39 = bitcast i64* %v163 to i8*
  %v164 = bitcast i8* %v39 to i64*
  store i64 %v38, i64* %v164, align 8
  %v165 = bitcast i8* %v39 to { i64 }*
  %v40 = load { i64 }, { i64 }* %v165, align 8
  %v41 = extractvalue { i64 } %v40, 0
  %v42 = sub i64 %v41, 0
  %v43 = icmp ule i64 %v42, 0
  %v44 = add i64 %v42, 0
  %v45 = select i1 %v43, i64 %v44, i64 1
  %v46 = icmp eq i64 %v45, 1
  %v166 = alloca { i64 }, align 8
  %v47 = bitcast { i64 }* %v166 to i8*
  %v167 = bitcast i8* %v47 to { i64 }*
  store { i64 } %v40, { i64 }* %v167, align 8
  %v48 = getelementptr inbounds i8, i8* %v47, i64 0
  %v168 = bitcast i8* %v48 to { { i64 } }*
  %v49 = load { { i64 } }, { { i64 } }* %v168, align 8
  %v169 = alloca { { i64 } }, align 8
  %v50 = bitcast { { i64 } }* %v169 to i8*
  %v170 = bitcast i8* %v50 to { { i64 } }*
  store { { i64 } } %v49, { { i64 } }* %v170, align 8
  %v171 = bitcast i8* %v50 to i64*
  %v51 = load i64, i64* %v171, align 8
  %v52 = icmp ugt i64 %v51, 4294967295
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v54 = extractvalue { i32, i32 } %v131, 1
  %v55 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v54) #0
  %v172 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v55, { i32, i32, i32, i32 }* %v172, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v173 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v174 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v173, i32 0, i32 0
  %v56 = bitcast i32* %v174 to i8*
  %v175 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v175, align 4
  %v176 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v177 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v176, i32 0, i32 1
  %v58 = bitcast i32* %v177 to i8*
  %v178 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v178, align 4
  %v179 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v180 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v179, i32 0, i32 2
  %v60 = bitcast i32* %v180 to i8*
  %v181 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v181, align 4
  %v182 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v183 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v182, i32 0, i32 3
  %v62 = bitcast i32* %v183 to i8*
  %v184 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v184, align 4
  %v185 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v186 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v185, i32 0, i32 4
  %v64 = bitcast i32* %v186 to i8*
  %v187 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v187, align 4
  %v66 = mul i32 %v57, %v65
  %v188 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v189 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v188, i32 0, i32 5
  %v67 = bitcast i32* %v189 to i8*
  %v190 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v190, align 4
  %v69 = mul i32 %v59, %v68
  %v70 = add i32 %v66, %v69
  %v191 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v192 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v191, i32 0, i32 6
  %v71 = bitcast i32* %v192 to i8*
  %v193 = bitcast i8* %v71 to i32*
  %v72 = load i32, i32* %v193, align 4
  %v73 = mul i32 %v61, %v72
  %v74 = add i32 %v70, %v73
  %v194 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v195 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v194, i32 0, i32 7
  %v75 = bitcast i32* %v195 to i8*
  %v196 = bitcast i8* %v75 to i32*
  %v76 = load i32, i32* %v196, align 4
  %v77 = mul i32 %v63, %v76
  %v78 = add i32 %v74, %v77
  %v79 = zext i32 %v78 to i64
  %v197 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v198 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v197, i32 0, i32 4
  %v80 = bitcast i32* %v198 to i8*
  %v199 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v199, align 4
  %v82 = mul i32 %v57, %v81
  %v200 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v201 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v200, i32 0, i32 5
  %v83 = bitcast i32* %v201 to i8*
  %v202 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v202, align 4
  %v85 = mul i32 %v59, %v84
  %v86 = add i32 %v82, %v85
  %v203 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v204 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v203, i32 0, i32 6
  %v87 = bitcast i32* %v204 to i8*
  %v205 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v205, align 4
  %v89 = mul i32 %v61, %v88
  %v90 = add i32 %v86, %v89
  %v206 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v207 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v206, i32 0, i32 7
  %v91 = bitcast i32* %v207 to i8*
  %v208 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v208, align 4
  %v93 = mul i32 %v63, %v92
  %v94 = add i32 %v90, %v93
  %v95 = zext i32 %v94 to i64
  %v96 = extractvalue { i8*, i64 } %v13, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { i8*, i64 } %v13, 0
  %v209 = bitcast i8* %v98 to float*
  %v210 = getelementptr inbounds float, float* %v209, i64 %v95
  %v99 = bitcast float* %v210 to i8*
  %v211 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v211, align 4
  %v101 = fmul contract float %v100, %v100
  %v102 = fsub contract float 1.0, %v101
  %v103 = extractvalue { i8*, i64 } %v12, 0
  %v212 = bitcast i8* %v103 to float*
  %v213 = getelementptr inbounds float, float* %v212, i64 %v79
  %v104 = bitcast float* %v213 to i8*
  %v214 = bitcast i8* %v104 to float*
  %v105 = load float, float* %v214, align 4
  %v106 = fmul contract float %v105, %v102
  %v215 = bitcast i8* %v104 to float*
  store float %v106, float* %v215, align 4
  br label %bb2
bb7:
  %v216 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v217 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v216, i32 0, i32 0
  %v107 = bitcast { i32, i32 }* %v217 to i8*
  %v218 = bitcast i8* %v107 to { i32, i32 }*
  %v219 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v218, i32 0, i32 0
  %v108 = bitcast i32* %v219 to i8*
  %v220 = bitcast i8* %v108 to i32*
  %v109 = load i32, i32* %v220, align 4
  %v221 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v222 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v221, i32 0, i32 0
  %v110 = bitcast { i32, i32 }* %v222 to i8*
  %v223 = bitcast i8* %v110 to { i32, i32 }*
  %v224 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v223, i32 0, i32 1
  %v111 = bitcast i32* %v224 to i8*
  %v225 = bitcast i8* %v111 to i32*
  %v112 = load i32, i32* %v225, align 4
  %v226 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v227 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v226, i32 0, i32 1
  %v113 = bitcast i64* %v227 to i8*
  %v228 = bitcast i8* %v113 to i64*
  %v114 = load i64, i64* %v228, align 8
  %v229 = bitcast i8* %v15 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v230 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v229, i32 0, i32 2
  %v115 = bitcast i1* %v230 to i8*
  %v231 = bitcast i8* %v115 to i1*
  %v116 = load i1, i1* %v231, align 1
  br label %bb2
bb8:
  %v117 = add i32 %v36, %v137
  %v118 = sub i32 %v37, 1
  %v119 = insertvalue { i32, i32 } undef, i32 1, 0
  %v120 = insertvalue { i32, i32 } %v119, i32 %v36, 1
  %v121 = extractvalue { i32, i32 } %v120, 0
  %v122 = extractvalue { i32, i32 } %v120, 1
  br label %bb10
bb9:
  %v123 = insertvalue { i32, i32 } undef, i32 0, 0
  %v124 = extractvalue { i32, i32 } %v123, 0
  %v125 = extractvalue { i32, i32 } %v123, 1
  br label %bb10
bb10:
  %v126 = phi i32 [ %v117, %bb8 ], [ %v36, %bb9 ]
  %v127 = phi i32 [ %v118, %bb8 ], [ %v37, %bb9 ]
  %v128 = phi i32 [ %v121, %bb8 ], [ %v124, %bb9 ]
  %v129 = phi i32 [ %v122, %bb8 ], [ %v125, %bb9 ]
  %v130 = insertvalue { i32, i32 } undef, i32 %v128, 0
  %v131 = insertvalue { i32, i32 } %v130, i32 %v129, 1
  %v132 = extractvalue { i32, i32 } %v131, 0
  %v133 = zext i32 %v132 to i64
  %v134 = icmp eq i64 %v133, 0
  br i1 %v134, label %bb5, label %bb11
bb11:
  %v135 = icmp eq i64 %v133, 1
  br i1 %v135, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v136 = trunc i64 %v51 to i32
  br label %bb14
bb14:
  %v137 = phi i32 [ 4294967295, %bb12 ], [ %v136, %bb13 ]
  %v138 = icmp ugt i32 %v37, 0
  %v139 = xor i1 %v138, 1
  br i1 %v139, label %bb9, label %bb8
}

define void @gpu_elu_vec4_cuda_entry_045fc3e50e2a11db(i8* %v0, i8* %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v4 = insertvalue { i8*, i64 } %v3, i64 %v2, 1
  br label %bb0
bb0:
  %v5 = phi i8* [ %v0, %entry ]
  %v6 = phi { i8*, i64 } [ %v4, %entry ]
  %v121 = alloca { i32, i32, i32 }, align 4
  %v7 = bitcast { i32, i32, i32 }* %v121 to i8*
  %v122 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v8 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v122 to i8*
  %v9 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v123 = bitcast i8* %v7 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v9, { i32, i32, i32 }* %v123, align 4
  br label %bb1
bb1:
  %v124 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v125 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v124, i32 0, i32 0
  %v10 = bitcast i32* %v125 to i8*
  %v126 = bitcast i8* %v10 to i32*
  %v11 = load i32, i32* %v126, align 4
  %v127 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v128 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v127, i32 0, i32 1
  %v12 = bitcast i32* %v128 to i8*
  %v129 = bitcast i8* %v12 to i32*
  %v13 = load i32, i32* %v129, align 4
  %v14 = mul i32 %v11, %v13
  %v130 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v131 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v130, i32 0, i32 2
  %v15 = bitcast i32* %v131 to i8*
  %v132 = bitcast i8* %v15 to i32*
  %v16 = load i32, i32* %v132, align 4
  %v17 = mul i32 %v14, %v16
  %v133 = bitcast i8* %v5 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v134 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v133, i32 0, i32 3
  %v18 = bitcast i32* %v134 to i8*
  %v135 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v135, align 4
  %v20 = mul i32 %v17, %v19
  %v21 = udiv i32 %v20, 4
  %v136 = bitcast i8* %v7 to { i32, i32, i32 }*
  %v137 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v136, i32 0, i32 0
  %v22 = bitcast i32* %v137 to i8*
  %v138 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v138, align 4
  %v24 = insertvalue { i32, i32 } undef, i32 %v23, 0
  %v25 = insertvalue { i32, i32 } %v24, i32 %v21, 1
  %v26 = extractvalue { i32, i32 } %v25, 0
  %v27 = extractvalue { i32, i32 } %v25, 1
  %v28 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v26, i32 %v27, i64 16776960) #0
  %v140 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v28, { { i32, i32 }, i64, i1, [7 x i8] }* %v140, align 8
  br label %bb6
bb2:
  %v29 = phi i32 [ %v65, %bb6 ], [ %v82, %bb28 ]
  %v30 = phi i32 [ %v68, %bb6 ], [ %v83, %bb28 ]
  %v31 = add i64 %v70, 1
  %v141 = alloca i64, align 8
  %v32 = bitcast i64* %v141 to i8*
  %v142 = bitcast i8* %v32 to i64*
  store i64 %v31, i64* %v142, align 8
  %v143 = bitcast i8* %v32 to { i64 }*
  %v33 = load { i64 }, { i64 }* %v143, align 8
  %v34 = extractvalue { i64 } %v33, 0
  %v35 = sub i64 %v34, 0
  %v36 = icmp ule i64 %v35, 0
  %v37 = add i64 %v35, 0
  %v38 = select i1 %v36, i64 %v37, i64 1
  %v39 = icmp eq i64 %v38, 1
  %v144 = alloca { i64 }, align 8
  %v40 = bitcast { i64 }* %v144 to i8*
  %v145 = bitcast i8* %v40 to { i64 }*
  store { i64 } %v33, { i64 }* %v145, align 8
  %v41 = getelementptr inbounds i8, i8* %v40, i64 0
  %v146 = bitcast i8* %v41 to { { i64 } }*
  %v42 = load { { i64 } }, { { i64 } }* %v146, align 8
  %v147 = alloca { { i64 } }, align 8
  %v43 = bitcast { { i64 } }* %v147 to i8*
  %v148 = bitcast i8* %v43 to { { i64 } }*
  store { { i64 } } %v42, { { i64 } }* %v148, align 8
  %v149 = bitcast i8* %v43 to i64*
  %v44 = load i64, i64* %v149, align 8
  %v45 = icmp ugt i64 %v44, 4294967295
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb12, label %bb11
bb3:
  unreachable
bb4:
  %v47 = extractvalue { i32, i32 } %v87, 1
  %v48 = zext i32 %v47 to i64
  %v49 = extractvalue { i8*, i64 } %v6, 1
  %v50 = icmp ult i64 %v48, %v49
  %v51 = extractvalue { i8*, i64 } %v6, 0
  %v150 = bitcast i8* %v51 to { float, float, float, float }*
  %v151 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v150, i64 %v48
  %v52 = bitcast { float, float, float, float }* %v151 to i8*
  %v152 = bitcast i8* %v52 to { float, float, float, float }*
  %v153 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v152, i32 0, i32 0
  %v53 = bitcast float* %v153 to i8*
  %v154 = bitcast i8* %v53 to float*
  %v54 = load float, float* %v154, align 4
  %v155 = bitcast i8* %v52 to { float, float, float, float }*
  %v156 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v155, i32 0, i32 1
  %v55 = bitcast float* %v156 to i8*
  %v157 = bitcast i8* %v55 to float*
  %v56 = load float, float* %v157, align 4
  %v158 = bitcast i8* %v52 to { float, float, float, float }*
  %v159 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v158, i32 0, i32 2
  %v57 = bitcast float* %v159 to i8*
  %v160 = bitcast i8* %v57 to float*
  %v58 = load float, float* %v160, align 4
  %v161 = bitcast i8* %v52 to { float, float, float, float }*
  %v162 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v161, i32 0, i32 3
  %v59 = bitcast float* %v162 to i8*
  %v163 = bitcast i8* %v59 to float*
  %v60 = load float, float* %v163, align 4
  %v61 = fcmp ogt float %v54, 0.0
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb15, label %bb14
bb5:
  ret void
bb6:
  %v164 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v165 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v164, i32 0, i32 0
  %v63 = bitcast { i32, i32 }* %v165 to i8*
  %v166 = bitcast i8* %v63 to { i32, i32 }*
  %v167 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v166, i32 0, i32 0
  %v64 = bitcast i32* %v167 to i8*
  %v168 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v168, align 4
  %v169 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v170 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v169, i32 0, i32 0
  %v66 = bitcast { i32, i32 }* %v170 to i8*
  %v171 = bitcast i8* %v66 to { i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v171, i32 0, i32 1
  %v67 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v173, align 4
  %v174 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v175 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v174, i32 0, i32 1
  %v69 = bitcast i64* %v175 to i8*
  %v176 = bitcast i8* %v69 to i64*
  %v70 = load i64, i64* %v176, align 8
  %v177 = bitcast i8* %v8 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v178 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v177, i32 0, i32 2
  %v71 = bitcast i1* %v178 to i8*
  %v179 = bitcast i8* %v71 to i1*
  %v72 = load i1, i1* %v179, align 1
  br label %bb2
bb7:
  %v73 = add i32 %v29, %v93
  %v74 = sub i32 %v30, 1
  %v75 = insertvalue { i32, i32 } undef, i32 1, 0
  %v76 = insertvalue { i32, i32 } %v75, i32 %v29, 1
  %v77 = extractvalue { i32, i32 } %v76, 0
  %v78 = extractvalue { i32, i32 } %v76, 1
  br label %bb9
bb8:
  %v79 = insertvalue { i32, i32 } undef, i32 0, 0
  %v80 = extractvalue { i32, i32 } %v79, 0
  %v81 = extractvalue { i32, i32 } %v79, 1
  br label %bb9
bb9:
  %v82 = phi i32 [ %v73, %bb7 ], [ %v29, %bb8 ]
  %v83 = phi i32 [ %v74, %bb7 ], [ %v30, %bb8 ]
  %v84 = phi i32 [ %v77, %bb7 ], [ %v80, %bb8 ]
  %v85 = phi i32 [ %v78, %bb7 ], [ %v81, %bb8 ]
  %v86 = insertvalue { i32, i32 } undef, i32 %v84, 0
  %v87 = insertvalue { i32, i32 } %v86, i32 %v85, 1
  %v88 = extractvalue { i32, i32 } %v87, 0
  %v89 = zext i32 %v88 to i64
  %v90 = icmp eq i64 %v89, 0
  br i1 %v90, label %bb5, label %bb10
bb10:
  %v91 = icmp eq i64 %v89, 1
  br i1 %v91, label %bb4, label %bb3
bb11:
  br label %bb13
bb12:
  %v92 = trunc i64 %v44 to i32
  br label %bb13
bb13:
  %v93 = phi i32 [ 4294967295, %bb11 ], [ %v92, %bb12 ]
  %v94 = icmp ugt i32 %v30, 0
  %v95 = xor i1 %v94, 1
  br i1 %v95, label %bb8, label %bb7
bb14:
  br label %bb16
bb15:
  %v96 = call float @__nv_expf(float %v54) #0
  br label %bb17
bb16:
  %v97 = phi float [ %v54, %bb14 ], [ %v100, %bb17 ]
  %v98 = fcmp ogt float %v56, 0.0
  %v99 = xor i1 %v98, 1
  br i1 %v99, label %bb19, label %bb18
bb17:
  %v100 = fsub contract float %v96, 1.0
  br label %bb16
bb18:
  br label %bb20
bb19:
  %v101 = call float @__nv_expf(float %v56) #0
  br label %bb21
bb20:
  %v102 = phi float [ %v56, %bb18 ], [ %v105, %bb21 ]
  %v103 = fcmp ogt float %v58, 0.0
  %v104 = xor i1 %v103, 1
  br i1 %v104, label %bb23, label %bb22
bb21:
  %v105 = fsub contract float %v101, 1.0
  br label %bb20
bb22:
  br label %bb24
bb23:
  %v106 = call float @__nv_expf(float %v58) #0
  br label %bb25
bb24:
  %v107 = phi float [ %v58, %bb22 ], [ %v110, %bb25 ]
  %v108 = fcmp ogt float %v60, 0.0
  %v109 = xor i1 %v108, 1
  br i1 %v109, label %bb27, label %bb26
bb25:
  %v110 = fsub contract float %v106, 1.0
  br label %bb24
bb26:
  br label %bb28
bb27:
  %v111 = call float @__nv_expf(float %v60) #0
  br label %bb29
bb28:
  %v112 = phi float [ %v60, %bb26 ], [ %v119, %bb29 ]
  %v113 = insertvalue { float, float, float, float } undef, float %v97, 0
  %v114 = insertvalue { float, float, float, float } %v113, float %v102, 1
  %v115 = insertvalue { float, float, float, float } %v114, float %v107, 2
  %v116 = insertvalue { float, float, float, float } %v115, float %v112, 3
  %v117 = extractvalue { i8*, i64 } %v6, 0
  %v184 = bitcast i8* %v117 to { float, float, float, float }*
  %v185 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v184, i64 %v48
  %v118 = bitcast { float, float, float, float }* %v185 to i8*
  %v186 = bitcast i8* %v118 to { float, float, float, float }*
  store { float, float, float, float } %v116, { float, float, float, float }* %v186, align 4
  br label %bb2
bb29:
  %v119 = fsub contract float %v111, 1.0
  br label %bb28
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.trap()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare float @__nv_fmaf(float, float, float)

define void @gemm_tiled_vec4_cuda_entry_f80a592a3ac0a3f3(i8* %v0, i8* %v1, i8* %v2, i8* %v3, i64 %v4, i8* %v5, i64 %v6, i8* %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v10 = insertvalue { i8*, i64 } %v9, i64 %v4, 1
  %v11 = insertvalue { i8*, i64 } undef, i8* %v5, 0
  %v12 = insertvalue { i8*, i64 } %v11, i64 %v6, 1
  %v13 = insertvalue { i8*, i64 } undef, i8* %v7, 0
  %v14 = insertvalue { i8*, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi i8* [ %v0, %entry ]
  %v16 = phi i8* [ %v1, %entry ]
  %v17 = phi i8* [ %v2, %entry ]
  %v18 = phi { i8*, i64 } [ %v10, %entry ]
  %v19 = phi { i8*, i64 } [ %v12, %entry ]
  %v20 = phi { i8*, i64 } [ %v14, %entry ]
  %v303 = alloca [4 x { float, float, float, float }], align 4
  %v21 = bitcast [4 x { float, float, float, float }]* %v303 to i8*
  %v304 = alloca [4 x float], align 4
  %v22 = bitcast [4 x float]* %v304 to i8*
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb15
bb1:
  %v24 = phi i32 [ 0, %bb20 ], [ %v297, %bb54 ]
  %v25 = icmp ult i32 %v24, %v115
  %v26 = xor i1 %v25, 1
  br i1 %v26, label %bb6, label %bb2
bb2:
  %v27 = mul i32 %v109, 4
  %v28 = udiv i32 %v27, 16
  %v29 = urem i32 %v27, 16
  %v30 = add i32 %v110, %v28
  %v31 = mul i32 %v30, %v115
  %v32 = add i32 %v31, %v24
  %v33 = add i32 %v32, %v29
  %v34 = udiv i32 %v33, 4
  %v35 = zext i32 %v34 to i64
  %v36 = extractvalue { i8*, i64 } %v19, 1
  %v37 = icmp ult i64 %v35, %v36
  %v38 = extractvalue { i8*, i64 } %v19, 0
  %v305 = bitcast i8* %v38 to { float, float, float, float }*
  %v306 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v305, i64 %v35
  %v39 = bitcast { float, float, float, float }* %v306 to i8*
  %v307 = bitcast i8* %v39 to { float, float, float, float }*
  %v308 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v307, i32 0, i32 0
  %v40 = bitcast float* %v308 to i8*
  %v309 = bitcast i8* %v40 to float*
  %v41 = load float, float* %v309, align 4
  %v310 = bitcast i8* %v39 to { float, float, float, float }*
  %v311 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v310, i32 0, i32 1
  %v42 = bitcast float* %v311 to i8*
  %v312 = bitcast i8* %v42 to float*
  %v43 = load float, float* %v312, align 4
  %v313 = bitcast i8* %v39 to { float, float, float, float }*
  %v314 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v313, i32 0, i32 2
  %v44 = bitcast float* %v314 to i8*
  %v315 = bitcast i8* %v44 to float*
  %v45 = load float, float* %v315, align 4
  %v316 = bitcast i8* %v39 to { float, float, float, float }*
  %v317 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v316, i32 0, i32 3
  %v46 = bitcast float* %v317 to i8*
  %v318 = bitcast i8* %v46 to float*
  %v47 = load float, float* %v318, align 4
  %v48 = mul i32 %v28, 17
  %v49 = add i32 %v48, %v29
  %v50 = zext i32 %v49 to i64
  %v319 = bitcast i8 addrspace(3)* %v104 to float addrspace(3)*
  %v320 = getelementptr inbounds float, float addrspace(3)* %v319, i64 %v50
  %v51 = bitcast float addrspace(3)* %v320 to i8 addrspace(3)*
  br label %bb21
bb3:
  %v52 = phi i32 [ 0, %bb29 ], [ %v296, %bb53 ]
  %v53 = icmp ult i32 %v52, 16
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb5, label %bb4
bb4:
  %v55 = mul i32 %v161, 17
  %v56 = add i32 %v55, %v52
  %v57 = zext i32 %v56 to i64
  %v58 = getelementptr i8, i8 addrspace(3)* %v104, i64 0
  %v321 = bitcast i8 addrspace(3)* %v58 to float addrspace(3)*
  %v322 = getelementptr inbounds float, float addrspace(3)* %v321, i64 %v57
  %v59 = bitcast float addrspace(3)* %v322 to i8 addrspace(3)*
  br label %bb30
bb5:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb54
bb6:
  %v61 = mul i32 %v99, 4
  %v62 = add i32 %v110, %v61
  %v63 = mul i32 %v23, 4
  %v64 = add i32 %v111, %v63
  br label %bb7
bb7:
  %v65 = phi i32 [ 0, %bb6 ], [ %v98, %bb13 ]
  %v66 = icmp ult i32 %v65, 4
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb14, label %bb8
bb8:
  %v68 = zext i32 %v65 to i64
  %v69 = icmp ult i64 %v68, 4
  br i1 %v69, label %bb9, label %bb55
bb9:
  %v323 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v324 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v323, i32 0, i64 %v68
  %v70 = bitcast { float, float, float, float }* %v324 to i8*
  %v325 = bitcast i8* %v70 to { float, float, float, float }*
  %v326 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v325, i32 0, i32 0
  %v71 = bitcast float* %v326 to i8*
  %v327 = bitcast i8* %v71 to float*
  %v72 = load float, float* %v327, align 4
  %v328 = bitcast i8* %v70 to { float, float, float, float }*
  %v329 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v328, i32 0, i32 1
  %v73 = bitcast float* %v329 to i8*
  %v330 = bitcast i8* %v73 to float*
  %v74 = load float, float* %v330, align 4
  %v331 = bitcast i8* %v70 to { float, float, float, float }*
  %v332 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v331, i32 0, i32 2
  %v75 = bitcast float* %v332 to i8*
  %v333 = bitcast i8* %v75 to float*
  %v76 = load float, float* %v333, align 4
  %v334 = bitcast i8* %v70 to { float, float, float, float }*
  %v335 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v334, i32 0, i32 3
  %v77 = bitcast float* %v335 to i8*
  %v336 = bitcast i8* %v77 to float*
  %v78 = load float, float* %v336, align 4
  %v337 = bitcast i8* %v22 to [4 x float]*
  %v338 = getelementptr inbounds [4 x float], [4 x float]* %v337, i32 0, i64 0
  %v79 = bitcast float* %v338 to i8*
  %v339 = bitcast i8* %v79 to float*
  store float %v72, float* %v339, align 4
  %v340 = bitcast i8* %v22 to [4 x float]*
  %v341 = getelementptr inbounds [4 x float], [4 x float]* %v340, i32 0, i64 1
  %v80 = bitcast float* %v341 to i8*
  %v342 = bitcast i8* %v80 to float*
  store float %v74, float* %v342, align 4
  %v343 = bitcast i8* %v22 to [4 x float]*
  %v344 = getelementptr inbounds [4 x float], [4 x float]* %v343, i32 0, i64 2
  %v81 = bitcast float* %v344 to i8*
  %v345 = bitcast i8* %v81 to float*
  store float %v76, float* %v345, align 4
  %v346 = bitcast i8* %v22 to [4 x float]*
  %v347 = getelementptr inbounds [4 x float], [4 x float]* %v346, i32 0, i64 3
  %v82 = bitcast float* %v347 to i8*
  %v348 = bitcast i8* %v82 to float*
  store float %v78, float* %v348, align 4
  br label %bb10
bb10:
  %v83 = phi i32 [ 0, %bb9 ], [ %v97, %bb12 ]
  %v84 = icmp ult i32 %v83, 4
  %v85 = xor i1 %v84, 1
  br i1 %v85, label %bb13, label %bb11
bb11:
  %v86 = add i32 %v62, %v65
  %v87 = mul i32 %v86, %v113
  %v88 = add i32 %v87, %v64
  %v89 = add i32 %v88, %v83
  %v90 = zext i32 %v89 to i64
  %v91 = zext i32 %v83 to i64
  %v92 = icmp ult i64 %v91, 4
  br i1 %v92, label %bb12, label %bb56
bb12:
  %v349 = bitcast i8* %v22 to [4 x float]*
  %v350 = getelementptr inbounds [4 x float], [4 x float]* %v349, i32 0, i64 %v91
  %v93 = bitcast float* %v350 to i8*
  %v351 = bitcast i8* %v93 to float*
  %v94 = load float, float* %v351, align 4
  %v95 = extractvalue { i8*, i64 } %v18, 0
  %v352 = bitcast i8* %v95 to float*
  %v353 = getelementptr inbounds float, float* %v352, i64 %v90
  %v96 = bitcast float* %v353 to i8*
  %v354 = bitcast i8* %v96 to float*
  store float %v94, float* %v354, align 4
  %v97 = add i32 %v83, 1
  br label %bb10
bb13:
  %v98 = add i32 %v65, 1
  br label %bb7
bb14:
  ret void
bb15:
  %v99 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb16
bb16:
  %v100 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb17
bb17:
  %v101 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb18
bb18:
  %v102 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb19
bb19:
  %v103 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #0
  br label %bb20
bb20:
  %v104 = bitcast [1088 x float] addrspace(3)* @__shared_mem_13 to i8 addrspace(3)*
  %v105 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v104, 0
  %v106 = bitcast [1040 x float] addrspace(3)* @__shared_mem_14 to i8 addrspace(3)*
  %v107 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v106, 0
  %v108 = mul i32 %v99, 16
  %v109 = add i32 %v108, %v23
  %v110 = mul i32 %v102, 64
  %v111 = mul i32 %v101, 64
  %v357 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v358 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v357, i32 0, i32 3
  %v112 = bitcast i32* %v358 to i8*
  %v359 = bitcast i8* %v112 to i32*
  %v113 = load i32, i32* %v359, align 4
  %v360 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v361 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v360, i32 0, i32 3
  %v114 = bitcast i32* %v361 to i8*
  %v362 = bitcast i8* %v114 to i32*
  %v115 = load i32, i32* %v362, align 4
  %v116 = insertvalue { float, float, float, float } undef, float 0.0, 0
  %v117 = insertvalue { float, float, float, float } %v116, float 0.0, 1
  %v118 = insertvalue { float, float, float, float } %v117, float 0.0, 2
  %v119 = insertvalue { float, float, float, float } %v118, float 0.0, 3
  %v364 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v365 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v364, i32 0, i64 0
  %v120 = bitcast { float, float, float, float }* %v365 to i8*
  %v366 = bitcast i8* %v120 to { float, float, float, float }*
  store { float, float, float, float } %v119, { float, float, float, float }* %v366, align 4
  %v367 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v368 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v367, i32 0, i64 1
  %v121 = bitcast { float, float, float, float }* %v368 to i8*
  %v369 = bitcast i8* %v121 to { float, float, float, float }*
  store { float, float, float, float } %v119, { float, float, float, float }* %v369, align 4
  %v370 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v371 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v370, i32 0, i64 2
  %v122 = bitcast { float, float, float, float }* %v371 to i8*
  %v372 = bitcast i8* %v122 to { float, float, float, float }*
  store { float, float, float, float } %v119, { float, float, float, float }* %v372, align 4
  %v373 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v374 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v373, i32 0, i64 3
  %v123 = bitcast { float, float, float, float }* %v374 to i8*
  %v375 = bitcast i8* %v123 to { float, float, float, float }*
  store { float, float, float, float } %v119, { float, float, float, float }* %v375, align 4
  br label %bb1
bb21:
  %v376 = bitcast i8 addrspace(3)* %v51 to float addrspace(3)*
  store float %v41, float addrspace(3)* %v376, align 4
  %v124 = add i64 %v50, 1
  %v377 = bitcast i8 addrspace(3)* %v104 to float addrspace(3)*
  %v378 = getelementptr inbounds float, float addrspace(3)* %v377, i64 %v124
  %v125 = bitcast float addrspace(3)* %v378 to i8 addrspace(3)*
  br label %bb22
bb22:
  %v379 = bitcast i8 addrspace(3)* %v125 to float addrspace(3)*
  store float %v43, float addrspace(3)* %v379, align 4
  %v126 = add i64 %v50, 2
  %v380 = bitcast i8 addrspace(3)* %v104 to float addrspace(3)*
  %v381 = getelementptr inbounds float, float addrspace(3)* %v380, i64 %v126
  %v127 = bitcast float addrspace(3)* %v381 to i8 addrspace(3)*
  br label %bb23
bb23:
  %v382 = bitcast i8 addrspace(3)* %v127 to float addrspace(3)*
  store float %v45, float addrspace(3)* %v382, align 4
  %v128 = add i64 %v50, 3
  %v383 = bitcast i8 addrspace(3)* %v104 to float addrspace(3)*
  %v384 = getelementptr inbounds float, float addrspace(3)* %v383, i64 %v128
  %v129 = bitcast float addrspace(3)* %v384 to i8 addrspace(3)*
  br label %bb24
bb24:
  %v385 = bitcast i8 addrspace(3)* %v129 to float addrspace(3)*
  store float %v47, float addrspace(3)* %v385, align 4
  %v130 = udiv i32 %v27, 64
  %v131 = urem i32 %v27, 64
  %v132 = add i32 %v24, %v130
  %v133 = mul i32 %v132, %v113
  %v134 = add i32 %v133, %v111
  %v135 = add i32 %v134, %v131
  %v136 = udiv i32 %v135, 4
  %v137 = zext i32 %v136 to i64
  %v138 = extractvalue { i8*, i64 } %v20, 1
  %v139 = icmp ult i64 %v137, %v138
  %v140 = extractvalue { i8*, i64 } %v20, 0
  %v386 = bitcast i8* %v140 to { float, float, float, float }*
  %v387 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v386, i64 %v137
  %v141 = bitcast { float, float, float, float }* %v387 to i8*
  %v388 = bitcast i8* %v141 to { float, float, float, float }*
  %v389 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v388, i32 0, i32 0
  %v142 = bitcast float* %v389 to i8*
  %v390 = bitcast i8* %v142 to float*
  %v143 = load float, float* %v390, align 4
  %v391 = bitcast i8* %v141 to { float, float, float, float }*
  %v392 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v391, i32 0, i32 1
  %v144 = bitcast float* %v392 to i8*
  %v393 = bitcast i8* %v144 to float*
  %v145 = load float, float* %v393, align 4
  %v394 = bitcast i8* %v141 to { float, float, float, float }*
  %v395 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v394, i32 0, i32 2
  %v146 = bitcast float* %v395 to i8*
  %v396 = bitcast i8* %v146 to float*
  %v147 = load float, float* %v396, align 4
  %v397 = bitcast i8* %v141 to { float, float, float, float }*
  %v398 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v397, i32 0, i32 3
  %v148 = bitcast float* %v398 to i8*
  %v399 = bitcast i8* %v148 to float*
  %v149 = load float, float* %v399, align 4
  %v150 = mul i32 %v130, 65
  %v151 = add i32 %v150, %v131
  %v152 = zext i32 %v151 to i64
  %v400 = bitcast i8 addrspace(3)* %v106 to float addrspace(3)*
  %v401 = getelementptr inbounds float, float addrspace(3)* %v400, i64 %v152
  %v153 = bitcast float addrspace(3)* %v401 to i8 addrspace(3)*
  br label %bb25
bb25:
  %v402 = bitcast i8 addrspace(3)* %v153 to float addrspace(3)*
  store float %v143, float addrspace(3)* %v402, align 4
  %v154 = add i64 %v152, 1
  %v403 = bitcast i8 addrspace(3)* %v106 to float addrspace(3)*
  %v404 = getelementptr inbounds float, float addrspace(3)* %v403, i64 %v154
  %v155 = bitcast float addrspace(3)* %v404 to i8 addrspace(3)*
  br label %bb26
bb26:
  %v405 = bitcast i8 addrspace(3)* %v155 to float addrspace(3)*
  store float %v145, float addrspace(3)* %v405, align 4
  %v156 = add i64 %v152, 2
  %v406 = bitcast i8 addrspace(3)* %v106 to float addrspace(3)*
  %v407 = getelementptr inbounds float, float addrspace(3)* %v406, i64 %v156
  %v157 = bitcast float addrspace(3)* %v407 to i8 addrspace(3)*
  br label %bb27
bb27:
  %v408 = bitcast i8 addrspace(3)* %v157 to float addrspace(3)*
  store float %v147, float addrspace(3)* %v408, align 4
  %v158 = add i64 %v152, 3
  %v409 = bitcast i8 addrspace(3)* %v106 to float addrspace(3)*
  %v410 = getelementptr inbounds float, float addrspace(3)* %v409, i64 %v158
  %v159 = bitcast float addrspace(3)* %v410 to i8 addrspace(3)*
  br label %bb28
bb28:
  %v411 = bitcast i8 addrspace(3)* %v159 to float addrspace(3)*
  store float %v149, float addrspace(3)* %v411, align 4
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb29
bb29:
  %v161 = mul i32 %v99, 4
  %v162 = mul i32 %v23, 4
  br label %bb3
bb30:
  %v412 = bitcast i8 addrspace(3)* %v59 to float addrspace(3)*
  %v163 = load float, float addrspace(3)* %v412, align 4
  %v164 = add i32 %v161, 1
  %v165 = mul i32 %v164, 17
  %v166 = add i32 %v165, %v52
  %v167 = zext i32 %v166 to i64
  %v168 = getelementptr i8, i8 addrspace(3)* %v104, i64 0
  %v413 = bitcast i8 addrspace(3)* %v168 to float addrspace(3)*
  %v414 = getelementptr inbounds float, float addrspace(3)* %v413, i64 %v167
  %v169 = bitcast float addrspace(3)* %v414 to i8 addrspace(3)*
  br label %bb31
bb31:
  %v415 = bitcast i8 addrspace(3)* %v169 to float addrspace(3)*
  %v170 = load float, float addrspace(3)* %v415, align 4
  %v171 = add i32 %v161, 2
  %v172 = mul i32 %v171, 17
  %v173 = add i32 %v172, %v52
  %v174 = zext i32 %v173 to i64
  %v175 = getelementptr i8, i8 addrspace(3)* %v104, i64 0
  %v416 = bitcast i8 addrspace(3)* %v175 to float addrspace(3)*
  %v417 = getelementptr inbounds float, float addrspace(3)* %v416, i64 %v174
  %v176 = bitcast float addrspace(3)* %v417 to i8 addrspace(3)*
  br label %bb32
bb32:
  %v418 = bitcast i8 addrspace(3)* %v176 to float addrspace(3)*
  %v177 = load float, float addrspace(3)* %v418, align 4
  %v178 = add i32 %v161, 3
  %v179 = mul i32 %v178, 17
  %v180 = add i32 %v179, %v52
  %v181 = zext i32 %v180 to i64
  %v182 = getelementptr i8, i8 addrspace(3)* %v104, i64 0
  %v419 = bitcast i8 addrspace(3)* %v182 to float addrspace(3)*
  %v420 = getelementptr inbounds float, float addrspace(3)* %v419, i64 %v181
  %v183 = bitcast float addrspace(3)* %v420 to i8 addrspace(3)*
  br label %bb33
bb33:
  %v421 = bitcast i8 addrspace(3)* %v183 to float addrspace(3)*
  %v184 = load float, float addrspace(3)* %v421, align 4
  %v185 = mul i32 %v52, 65
  %v186 = add i32 %v185, %v162
  %v187 = zext i32 %v186 to i64
  %v188 = getelementptr i8, i8 addrspace(3)* %v106, i64 0
  %v422 = bitcast i8 addrspace(3)* %v188 to float addrspace(3)*
  %v423 = getelementptr inbounds float, float addrspace(3)* %v422, i64 %v187
  %v189 = bitcast float addrspace(3)* %v423 to i8 addrspace(3)*
  br label %bb34
bb34:
  %v424 = bitcast i8 addrspace(3)* %v189 to float addrspace(3)*
  %v190 = load float, float addrspace(3)* %v424, align 4
  %v191 = mul i32 %v52, 65
  %v192 = add i32 %v191, %v162
  %v193 = add i32 %v192, 1
  %v194 = zext i32 %v193 to i64
  %v195 = getelementptr i8, i8 addrspace(3)* %v106, i64 0
  %v425 = bitcast i8 addrspace(3)* %v195 to float addrspace(3)*
  %v426 = getelementptr inbounds float, float addrspace(3)* %v425, i64 %v194
  %v196 = bitcast float addrspace(3)* %v426 to i8 addrspace(3)*
  br label %bb35
bb35:
  %v427 = bitcast i8 addrspace(3)* %v196 to float addrspace(3)*
  %v197 = load float, float addrspace(3)* %v427, align 4
  %v198 = mul i32 %v52, 65
  %v199 = add i32 %v198, %v162
  %v200 = add i32 %v199, 2
  %v201 = zext i32 %v200 to i64
  %v202 = getelementptr i8, i8 addrspace(3)* %v106, i64 0
  %v428 = bitcast i8 addrspace(3)* %v202 to float addrspace(3)*
  %v429 = getelementptr inbounds float, float addrspace(3)* %v428, i64 %v201
  %v203 = bitcast float addrspace(3)* %v429 to i8 addrspace(3)*
  br label %bb36
bb36:
  %v430 = bitcast i8 addrspace(3)* %v203 to float addrspace(3)*
  %v204 = load float, float addrspace(3)* %v430, align 4
  %v205 = mul i32 %v52, 65
  %v206 = add i32 %v205, %v162
  %v207 = add i32 %v206, 3
  %v208 = zext i32 %v207 to i64
  %v209 = getelementptr i8, i8 addrspace(3)* %v106, i64 0
  %v431 = bitcast i8 addrspace(3)* %v209 to float addrspace(3)*
  %v432 = getelementptr inbounds float, float addrspace(3)* %v431, i64 %v208
  %v210 = bitcast float addrspace(3)* %v432 to i8 addrspace(3)*
  br label %bb37
bb37:
  %v433 = bitcast i8 addrspace(3)* %v210 to float addrspace(3)*
  %v211 = load float, float addrspace(3)* %v433, align 4
  %v434 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v435 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v434, i32 0, i64 0
  %v212 = bitcast { float, float, float, float }* %v435 to i8*
  %v436 = bitcast i8* %v212 to { float, float, float, float }*
  %v437 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v436, i32 0, i32 0
  %v213 = bitcast float* %v437 to i8*
  %v438 = bitcast i8* %v213 to float*
  %v214 = load float, float* %v438, align 4
  %v439 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v440 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v439, i32 0, i64 0
  %v215 = bitcast { float, float, float, float }* %v440 to i8*
  %v441 = bitcast i8* %v215 to { float, float, float, float }*
  %v442 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v441, i32 0, i32 1
  %v216 = bitcast float* %v442 to i8*
  %v443 = bitcast i8* %v216 to float*
  %v217 = load float, float* %v443, align 4
  %v444 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v445 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v444, i32 0, i64 0
  %v218 = bitcast { float, float, float, float }* %v445 to i8*
  %v446 = bitcast i8* %v218 to { float, float, float, float }*
  %v447 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v446, i32 0, i32 2
  %v219 = bitcast float* %v447 to i8*
  %v448 = bitcast i8* %v219 to float*
  %v220 = load float, float* %v448, align 4
  %v449 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v450 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v449, i32 0, i64 0
  %v221 = bitcast { float, float, float, float }* %v450 to i8*
  %v451 = bitcast i8* %v221 to { float, float, float, float }*
  %v452 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v451, i32 0, i32 3
  %v222 = bitcast float* %v452 to i8*
  %v453 = bitcast i8* %v222 to float*
  %v223 = load float, float* %v453, align 4
  %v224 = call float @__nv_fmaf(float %v190, float %v163, float %v214) #0
  br label %bb38
bb38:
  %v225 = call float @__nv_fmaf(float %v197, float %v163, float %v217) #0
  br label %bb39
bb39:
  %v226 = call float @__nv_fmaf(float %v204, float %v163, float %v220) #0
  br label %bb40
bb40:
  %v227 = call float @__nv_fmaf(float %v211, float %v163, float %v223) #0
  br label %bb41
bb41:
  %v228 = insertvalue { float, float, float, float } undef, float %v224, 0
  %v229 = insertvalue { float, float, float, float } %v228, float %v225, 1
  %v230 = insertvalue { float, float, float, float } %v229, float %v226, 2
  %v231 = insertvalue { float, float, float, float } %v230, float %v227, 3
  %v455 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v456 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v455, i32 0, i64 0
  %v232 = bitcast { float, float, float, float }* %v456 to i8*
  %v457 = bitcast i8* %v232 to { float, float, float, float }*
  store { float, float, float, float } %v231, { float, float, float, float }* %v457, align 4
  %v458 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v459 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v458, i32 0, i64 1
  %v233 = bitcast { float, float, float, float }* %v459 to i8*
  %v460 = bitcast i8* %v233 to { float, float, float, float }*
  %v461 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v460, i32 0, i32 0
  %v234 = bitcast float* %v461 to i8*
  %v462 = bitcast i8* %v234 to float*
  %v235 = load float, float* %v462, align 4
  %v463 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v464 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v463, i32 0, i64 1
  %v236 = bitcast { float, float, float, float }* %v464 to i8*
  %v465 = bitcast i8* %v236 to { float, float, float, float }*
  %v466 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v465, i32 0, i32 1
  %v237 = bitcast float* %v466 to i8*
  %v467 = bitcast i8* %v237 to float*
  %v238 = load float, float* %v467, align 4
  %v468 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v469 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v468, i32 0, i64 1
  %v239 = bitcast { float, float, float, float }* %v469 to i8*
  %v470 = bitcast i8* %v239 to { float, float, float, float }*
  %v471 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v470, i32 0, i32 2
  %v240 = bitcast float* %v471 to i8*
  %v472 = bitcast i8* %v240 to float*
  %v241 = load float, float* %v472, align 4
  %v473 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v474 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v473, i32 0, i64 1
  %v242 = bitcast { float, float, float, float }* %v474 to i8*
  %v475 = bitcast i8* %v242 to { float, float, float, float }*
  %v476 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v475, i32 0, i32 3
  %v243 = bitcast float* %v476 to i8*
  %v477 = bitcast i8* %v243 to float*
  %v244 = load float, float* %v477, align 4
  %v245 = call float @__nv_fmaf(float %v190, float %v170, float %v235) #0
  br label %bb42
bb42:
  %v246 = call float @__nv_fmaf(float %v197, float %v170, float %v238) #0
  br label %bb43
bb43:
  %v247 = call float @__nv_fmaf(float %v204, float %v170, float %v241) #0
  br label %bb44
bb44:
  %v248 = call float @__nv_fmaf(float %v211, float %v170, float %v244) #0
  br label %bb45
bb45:
  %v249 = insertvalue { float, float, float, float } undef, float %v245, 0
  %v250 = insertvalue { float, float, float, float } %v249, float %v246, 1
  %v251 = insertvalue { float, float, float, float } %v250, float %v247, 2
  %v252 = insertvalue { float, float, float, float } %v251, float %v248, 3
  %v479 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v480 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v479, i32 0, i64 1
  %v253 = bitcast { float, float, float, float }* %v480 to i8*
  %v481 = bitcast i8* %v253 to { float, float, float, float }*
  store { float, float, float, float } %v252, { float, float, float, float }* %v481, align 4
  %v482 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v483 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v482, i32 0, i64 2
  %v254 = bitcast { float, float, float, float }* %v483 to i8*
  %v484 = bitcast i8* %v254 to { float, float, float, float }*
  %v485 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v484, i32 0, i32 0
  %v255 = bitcast float* %v485 to i8*
  %v486 = bitcast i8* %v255 to float*
  %v256 = load float, float* %v486, align 4
  %v487 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v488 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v487, i32 0, i64 2
  %v257 = bitcast { float, float, float, float }* %v488 to i8*
  %v489 = bitcast i8* %v257 to { float, float, float, float }*
  %v490 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v489, i32 0, i32 1
  %v258 = bitcast float* %v490 to i8*
  %v491 = bitcast i8* %v258 to float*
  %v259 = load float, float* %v491, align 4
  %v492 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v493 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v492, i32 0, i64 2
  %v260 = bitcast { float, float, float, float }* %v493 to i8*
  %v494 = bitcast i8* %v260 to { float, float, float, float }*
  %v495 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v494, i32 0, i32 2
  %v261 = bitcast float* %v495 to i8*
  %v496 = bitcast i8* %v261 to float*
  %v262 = load float, float* %v496, align 4
  %v497 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v498 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v497, i32 0, i64 2
  %v263 = bitcast { float, float, float, float }* %v498 to i8*
  %v499 = bitcast i8* %v263 to { float, float, float, float }*
  %v500 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v499, i32 0, i32 3
  %v264 = bitcast float* %v500 to i8*
  %v501 = bitcast i8* %v264 to float*
  %v265 = load float, float* %v501, align 4
  %v266 = call float @__nv_fmaf(float %v190, float %v177, float %v256) #0
  br label %bb46
bb46:
  %v267 = call float @__nv_fmaf(float %v197, float %v177, float %v259) #0
  br label %bb47
bb47:
  %v268 = call float @__nv_fmaf(float %v204, float %v177, float %v262) #0
  br label %bb48
bb48:
  %v269 = call float @__nv_fmaf(float %v211, float %v177, float %v265) #0
  br label %bb49
bb49:
  %v270 = insertvalue { float, float, float, float } undef, float %v266, 0
  %v271 = insertvalue { float, float, float, float } %v270, float %v267, 1
  %v272 = insertvalue { float, float, float, float } %v271, float %v268, 2
  %v273 = insertvalue { float, float, float, float } %v272, float %v269, 3
  %v503 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v504 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v503, i32 0, i64 2
  %v274 = bitcast { float, float, float, float }* %v504 to i8*
  %v505 = bitcast i8* %v274 to { float, float, float, float }*
  store { float, float, float, float } %v273, { float, float, float, float }* %v505, align 4
  %v506 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v507 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v506, i32 0, i64 3
  %v275 = bitcast { float, float, float, float }* %v507 to i8*
  %v508 = bitcast i8* %v275 to { float, float, float, float }*
  %v509 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v508, i32 0, i32 0
  %v276 = bitcast float* %v509 to i8*
  %v510 = bitcast i8* %v276 to float*
  %v277 = load float, float* %v510, align 4
  %v511 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v512 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v511, i32 0, i64 3
  %v278 = bitcast { float, float, float, float }* %v512 to i8*
  %v513 = bitcast i8* %v278 to { float, float, float, float }*
  %v514 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v513, i32 0, i32 1
  %v279 = bitcast float* %v514 to i8*
  %v515 = bitcast i8* %v279 to float*
  %v280 = load float, float* %v515, align 4
  %v516 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v517 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v516, i32 0, i64 3
  %v281 = bitcast { float, float, float, float }* %v517 to i8*
  %v518 = bitcast i8* %v281 to { float, float, float, float }*
  %v519 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v518, i32 0, i32 2
  %v282 = bitcast float* %v519 to i8*
  %v520 = bitcast i8* %v282 to float*
  %v283 = load float, float* %v520, align 4
  %v521 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v522 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v521, i32 0, i64 3
  %v284 = bitcast { float, float, float, float }* %v522 to i8*
  %v523 = bitcast i8* %v284 to { float, float, float, float }*
  %v524 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v523, i32 0, i32 3
  %v285 = bitcast float* %v524 to i8*
  %v525 = bitcast i8* %v285 to float*
  %v286 = load float, float* %v525, align 4
  %v287 = call float @__nv_fmaf(float %v190, float %v184, float %v277) #0
  br label %bb50
bb50:
  %v288 = call float @__nv_fmaf(float %v197, float %v184, float %v280) #0
  br label %bb51
bb51:
  %v289 = call float @__nv_fmaf(float %v204, float %v184, float %v283) #0
  br label %bb52
bb52:
  %v290 = call float @__nv_fmaf(float %v211, float %v184, float %v286) #0
  br label %bb53
bb53:
  %v291 = insertvalue { float, float, float, float } undef, float %v287, 0
  %v292 = insertvalue { float, float, float, float } %v291, float %v288, 1
  %v293 = insertvalue { float, float, float, float } %v292, float %v289, 2
  %v294 = insertvalue { float, float, float, float } %v293, float %v290, 3
  %v527 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v528 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v527, i32 0, i64 3
  %v295 = bitcast { float, float, float, float }* %v528 to i8*
  %v529 = bitcast i8* %v295 to { float, float, float, float }*
  store { float, float, float, float } %v294, { float, float, float, float }* %v529, align 4
  %v296 = add i32 %v52, 1
  br label %bb3
bb54:
  %v297 = add i32 %v24, 16
  br label %bb1
bb55:
  call void @llvm.trap() #0
  unreachable
bb56:
  call void @llvm.trap() #0
  unreachable
}

define void @gemm_tiled_cuda_entry_21aa8599370d695c(i8* %v0, i8* %v1, i8* %v2, i8* %v3, i64 %v4, i8* %v5, i64 %v6, i8* %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v10 = insertvalue { i8*, i64 } %v9, i64 %v4, 1
  %v11 = insertvalue { i8*, i64 } undef, i8* %v5, 0
  %v12 = insertvalue { i8*, i64 } %v11, i64 %v6, 1
  %v13 = insertvalue { i8*, i64 } undef, i8* %v7, 0
  %v14 = insertvalue { i8*, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi i8* [ %v0, %entry ]
  %v16 = phi i8* [ %v1, %entry ]
  %v17 = phi i8* [ %v2, %entry ]
  %v18 = phi { i8*, i64 } [ %v10, %entry ]
  %v19 = phi { i8*, i64 } [ %v12, %entry ]
  %v20 = phi { i8*, i64 } [ %v14, %entry ]
  %v367 = alloca [4 x { float, float, float, float }], align 4
  %v21 = bitcast [4 x { float, float, float, float }]* %v367 to i8*
  %v368 = alloca [4 x float], align 4
  %v22 = bitcast [4 x float]* %v368 to i8*
  %v23 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  br label %bb1
bb1:
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb39
bb2:
  %v25 = phi i32 [ %v213, %bb38 ], [ 0, %bb44 ]
  %v26 = icmp ult i32 %v25, %v204
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb46, label %bb45
bb3:
  unreachable
bb4:
  %v28 = extractvalue { i32, i32 } %v217, 1
  %v369 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v370 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v369, i32 0, i32 1
  %v29 = bitcast i32* %v370 to i8*
  %v371 = bitcast i8* %v29 to i32*
  %v30 = load i32, i32* %v371, align 4
  %v31 = icmp eq i32 %v30, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb6, label %bb77
bb5:
  ret void
bb6:
  %v33 = urem i32 %v188, %v30
  %v34 = insertvalue { float, float, float, float } undef, float 0.0, 0
  %v35 = insertvalue { float, float, float, float } %v34, float 0.0, 1
  %v36 = insertvalue { float, float, float, float } %v35, float 0.0, 2
  %v37 = insertvalue { float, float, float, float } %v36, float 0.0, 3
  %v373 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v374 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v373, i32 0, i64 0
  %v38 = bitcast { float, float, float, float }* %v374 to i8*
  %v375 = bitcast i8* %v38 to { float, float, float, float }*
  store { float, float, float, float } %v37, { float, float, float, float }* %v375, align 4
  %v376 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v377 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v376, i32 0, i64 1
  %v39 = bitcast { float, float, float, float }* %v377 to i8*
  %v378 = bitcast i8* %v39 to { float, float, float, float }*
  store { float, float, float, float } %v37, { float, float, float, float }* %v378, align 4
  %v379 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v380 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v379, i32 0, i64 2
  %v40 = bitcast { float, float, float, float }* %v380 to i8*
  %v381 = bitcast i8* %v40 to { float, float, float, float }*
  store { float, float, float, float } %v37, { float, float, float, float }* %v381, align 4
  %v382 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v383 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v382, i32 0, i64 3
  %v41 = bitcast { float, float, float, float }* %v383 to i8*
  %v384 = bitcast i8* %v41 to { float, float, float, float }*
  store { float, float, float, float } %v37, { float, float, float, float }* %v384, align 4
  br label %bb7
bb7:
  %v42 = phi i32 [ 0, %bb6 ], [ %v360, %bb76 ]
  %v43 = icmp ult i32 %v42, %v202
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb26, label %bb8
bb8:
  br label %bb9
bb9:
  %v45 = phi i32 [ %v194, %bb8 ], [ %v222, %bb49 ]
  %v46 = icmp ult i32 %v45, 1024
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb15, label %bb10
bb10:
  %v48 = udiv i32 %v45, 16
  %v49 = urem i32 %v45, 16
  %v50 = add i32 %v195, %v48
  %v51 = add i32 %v42, %v49
  %v52 = icmp ult i32 %v50, %v198
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb13, label %bb11
bb11:
  %v54 = icmp ult i32 %v51, %v202
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb13, label %bb12
bb12:
  %v385 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v386 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v385, i32 0, i32 4
  %v56 = bitcast i32* %v386 to i8*
  %v387 = bitcast i8* %v56 to i32*
  %v57 = load i32, i32* %v387, align 4
  %v58 = mul i32 %v28, %v57
  %v388 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v389 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v388, i32 0, i32 5
  %v59 = bitcast i32* %v389 to i8*
  %v390 = bitcast i8* %v59 to i32*
  %v60 = load i32, i32* %v390, align 4
  %v61 = mul i32 %v33, %v60
  %v62 = add i32 %v58, %v61
  %v391 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v392 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v391, i32 0, i32 6
  %v63 = bitcast i32* %v392 to i8*
  %v393 = bitcast i8* %v63 to i32*
  %v64 = load i32, i32* %v393, align 4
  %v65 = mul i32 %v50, %v64
  %v66 = add i32 %v62, %v65
  %v394 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v395 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v394, i32 0, i32 7
  %v67 = bitcast i32* %v395 to i8*
  %v396 = bitcast i8* %v67 to i32*
  %v68 = load i32, i32* %v396, align 4
  %v69 = mul i32 %v51, %v68
  %v70 = add i32 %v66, %v69
  %v71 = zext i32 %v70 to i64
  %v72 = extractvalue { i8*, i64 } %v19, 1
  %v73 = icmp ult i64 %v71, %v72
  %v74 = extractvalue { i8*, i64 } %v19, 0
  %v397 = bitcast i8* %v74 to float*
  %v398 = getelementptr inbounds float, float* %v397, i64 %v71
  %v75 = bitcast float* %v398 to i8*
  %v399 = bitcast i8* %v75 to float*
  %v76 = load float, float* %v399, align 4
  br label %bb14
bb13:
  br label %bb14
bb14:
  %v77 = phi float [ %v76, %bb12 ], [ 0.0, %bb13 ]
  %v78 = mul i32 %v48, 17
  %v79 = add i32 %v78, %v49
  %v80 = zext i32 %v79 to i64
  %v400 = bitcast i8 addrspace(3)* %v189 to float addrspace(3)*
  %v401 = getelementptr inbounds float, float addrspace(3)* %v400, i64 %v80
  %v81 = bitcast float addrspace(3)* %v401 to i8 addrspace(3)*
  br label %bb49
bb15:
  br label %bb16
bb16:
  %v82 = phi i32 [ %v194, %bb15 ], [ %v223, %bb50 ]
  %v83 = icmp ult i32 %v82, 1024
  %v84 = xor i1 %v83, 1
  br i1 %v84, label %bb22, label %bb17
bb17:
  %v85 = udiv i32 %v82, 64
  %v86 = urem i32 %v82, 64
  %v87 = add i32 %v42, %v85
  %v88 = add i32 %v196, %v86
  %v89 = icmp ult i32 %v87, %v202
  %v90 = xor i1 %v89, 1
  br i1 %v90, label %bb20, label %bb18
bb18:
  %v91 = icmp ult i32 %v88, %v200
  %v92 = xor i1 %v91, 1
  br i1 %v92, label %bb20, label %bb19
bb19:
  %v402 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v403 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v402, i32 0, i32 4
  %v93 = bitcast i32* %v403 to i8*
  %v404 = bitcast i8* %v93 to i32*
  %v94 = load i32, i32* %v404, align 4
  %v95 = mul i32 %v28, %v94
  %v405 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v406 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v405, i32 0, i32 5
  %v96 = bitcast i32* %v406 to i8*
  %v407 = bitcast i8* %v96 to i32*
  %v97 = load i32, i32* %v407, align 4
  %v98 = mul i32 %v33, %v97
  %v99 = add i32 %v95, %v98
  %v408 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v409 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v408, i32 0, i32 6
  %v100 = bitcast i32* %v409 to i8*
  %v410 = bitcast i8* %v100 to i32*
  %v101 = load i32, i32* %v410, align 4
  %v102 = mul i32 %v87, %v101
  %v103 = add i32 %v99, %v102
  %v411 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v412 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v411, i32 0, i32 7
  %v104 = bitcast i32* %v412 to i8*
  %v413 = bitcast i8* %v104 to i32*
  %v105 = load i32, i32* %v413, align 4
  %v106 = mul i32 %v88, %v105
  %v107 = add i32 %v103, %v106
  %v108 = zext i32 %v107 to i64
  %v109 = extractvalue { i8*, i64 } %v20, 1
  %v110 = icmp ult i64 %v108, %v109
  %v111 = extractvalue { i8*, i64 } %v20, 0
  %v414 = bitcast i8* %v111 to float*
  %v415 = getelementptr inbounds float, float* %v414, i64 %v108
  %v112 = bitcast float* %v415 to i8*
  %v416 = bitcast i8* %v112 to float*
  %v113 = load float, float* %v416, align 4
  br label %bb21
bb20:
  br label %bb21
bb21:
  %v114 = phi float [ %v113, %bb19 ], [ 0.0, %bb20 ]
  %v115 = mul i32 %v85, 65
  %v116 = add i32 %v115, %v86
  %v117 = zext i32 %v116 to i64
  %v417 = bitcast i8 addrspace(3)* %v191 to float addrspace(3)*
  %v418 = getelementptr inbounds float, float addrspace(3)* %v417, i64 %v117
  %v118 = bitcast float addrspace(3)* %v418 to i8 addrspace(3)*
  br label %bb50
bb22:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb51
bb23:
  %v120 = phi i32 [ 0, %bb51 ], [ %v359, %bb75 ]
  %v121 = icmp ult i32 %v120, 16
  %v122 = xor i1 %v121, 1
  br i1 %v122, label %bb25, label %bb24
bb24:
  %v123 = mul i32 %v224, 17
  %v124 = add i32 %v123, %v120
  %v125 = zext i32 %v124 to i64
  %v126 = getelementptr i8, i8 addrspace(3)* %v189, i64 0
  %v419 = bitcast i8 addrspace(3)* %v126 to float addrspace(3)*
  %v420 = getelementptr inbounds float, float addrspace(3)* %v419, i64 %v125
  %v127 = bitcast float addrspace(3)* %v420 to i8 addrspace(3)*
  br label %bb52
bb25:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb76
bb26:
  %v129 = mul i32 %v184, 4
  %v130 = add i32 %v195, %v129
  %v131 = mul i32 %v24, 4
  %v132 = add i32 %v196, %v131
  br label %bb27
bb27:
  %v133 = phi i32 [ 0, %bb26 ], [ %v183, %bb37 ]
  %v134 = icmp ult i32 %v133, 4
  %v135 = xor i1 %v134, 1
  br i1 %v135, label %bb38, label %bb28
bb28:
  %v136 = add i32 %v130, %v133
  %v137 = icmp ult i32 %v136, %v198
  %v138 = xor i1 %v137, 1
  br i1 %v138, label %bb37, label %bb29
bb29:
  %v139 = zext i32 %v133 to i64
  %v140 = icmp ult i64 %v139, 4
  br i1 %v140, label %bb30, label %bb78
bb30:
  %v421 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v422 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v421, i32 0, i64 %v139
  %v141 = bitcast { float, float, float, float }* %v422 to i8*
  %v423 = bitcast i8* %v141 to { float, float, float, float }*
  %v424 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v423, i32 0, i32 0
  %v142 = bitcast float* %v424 to i8*
  %v425 = bitcast i8* %v142 to float*
  %v143 = load float, float* %v425, align 4
  %v426 = bitcast i8* %v141 to { float, float, float, float }*
  %v427 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v426, i32 0, i32 1
  %v144 = bitcast float* %v427 to i8*
  %v428 = bitcast i8* %v144 to float*
  %v145 = load float, float* %v428, align 4
  %v429 = bitcast i8* %v141 to { float, float, float, float }*
  %v430 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v429, i32 0, i32 2
  %v146 = bitcast float* %v430 to i8*
  %v431 = bitcast i8* %v146 to float*
  %v147 = load float, float* %v431, align 4
  %v432 = bitcast i8* %v141 to { float, float, float, float }*
  %v433 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v432, i32 0, i32 3
  %v148 = bitcast float* %v433 to i8*
  %v434 = bitcast i8* %v148 to float*
  %v149 = load float, float* %v434, align 4
  %v435 = bitcast i8* %v22 to [4 x float]*
  %v436 = getelementptr inbounds [4 x float], [4 x float]* %v435, i32 0, i64 0
  %v150 = bitcast float* %v436 to i8*
  %v437 = bitcast i8* %v150 to float*
  store float %v143, float* %v437, align 4
  %v438 = bitcast i8* %v22 to [4 x float]*
  %v439 = getelementptr inbounds [4 x float], [4 x float]* %v438, i32 0, i64 1
  %v151 = bitcast float* %v439 to i8*
  %v440 = bitcast i8* %v151 to float*
  store float %v145, float* %v440, align 4
  %v441 = bitcast i8* %v22 to [4 x float]*
  %v442 = getelementptr inbounds [4 x float], [4 x float]* %v441, i32 0, i64 2
  %v152 = bitcast float* %v442 to i8*
  %v443 = bitcast i8* %v152 to float*
  store float %v147, float* %v443, align 4
  %v444 = bitcast i8* %v22 to [4 x float]*
  %v445 = getelementptr inbounds [4 x float], [4 x float]* %v444, i32 0, i64 3
  %v153 = bitcast float* %v445 to i8*
  %v446 = bitcast i8* %v153 to float*
  store float %v149, float* %v446, align 4
  br label %bb31
bb31:
  %v154 = phi i32 [ 0, %bb30 ], [ %v182, %bb35 ]
  %v155 = icmp ult i32 %v154, 4
  %v156 = xor i1 %v155, 1
  br i1 %v156, label %bb36, label %bb32
bb32:
  %v157 = add i32 %v132, %v154
  %v158 = icmp ult i32 %v157, %v200
  %v159 = xor i1 %v158, 1
  br i1 %v159, label %bb35, label %bb33
bb33:
  %v447 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v448 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v447, i32 0, i32 4
  %v160 = bitcast i32* %v448 to i8*
  %v449 = bitcast i8* %v160 to i32*
  %v161 = load i32, i32* %v449, align 4
  %v162 = mul i32 %v28, %v161
  %v450 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v451 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v450, i32 0, i32 5
  %v163 = bitcast i32* %v451 to i8*
  %v452 = bitcast i8* %v163 to i32*
  %v164 = load i32, i32* %v452, align 4
  %v165 = mul i32 %v33, %v164
  %v166 = add i32 %v162, %v165
  %v453 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v454 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v453, i32 0, i32 6
  %v167 = bitcast i32* %v454 to i8*
  %v455 = bitcast i8* %v167 to i32*
  %v168 = load i32, i32* %v455, align 4
  %v169 = mul i32 %v136, %v168
  %v170 = add i32 %v166, %v169
  %v456 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v457 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v456, i32 0, i32 7
  %v171 = bitcast i32* %v457 to i8*
  %v458 = bitcast i8* %v171 to i32*
  %v172 = load i32, i32* %v458, align 4
  %v173 = mul i32 %v157, %v172
  %v174 = add i32 %v170, %v173
  %v175 = zext i32 %v174 to i64
  %v176 = zext i32 %v154 to i64
  %v177 = icmp ult i64 %v176, 4
  br i1 %v177, label %bb34, label %bb79
bb34:
  %v459 = bitcast i8* %v22 to [4 x float]*
  %v460 = getelementptr inbounds [4 x float], [4 x float]* %v459, i32 0, i64 %v176
  %v178 = bitcast float* %v460 to i8*
  %v461 = bitcast i8* %v178 to float*
  %v179 = load float, float* %v461, align 4
  %v180 = extractvalue { i8*, i64 } %v18, 0
  %v462 = bitcast i8* %v180 to float*
  %v463 = getelementptr inbounds float, float* %v462, i64 %v175
  %v181 = bitcast float* %v463 to i8*
  %v464 = bitcast i8* %v181 to float*
  store float %v179, float* %v464, align 4
  br label %bb35
bb35:
  %v182 = add i32 %v154, 1
  br label %bb31
bb36:
  br label %bb37
bb37:
  %v183 = add i32 %v133, 1
  br label %bb27
bb38:
  br label %bb2
bb39:
  %v184 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb40
bb40:
  %v185 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb41
bb41:
  %v186 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb42
bb42:
  %v187 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb43
bb43:
  %v188 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #0
  br label %bb44
bb44:
  %v189 = bitcast [1088 x float] addrspace(3)* @__shared_mem_15 to i8 addrspace(3)*
  %v190 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v189, 0
  %v191 = bitcast [1040 x float] addrspace(3)* @__shared_mem_16 to i8 addrspace(3)*
  %v192 = insertvalue { i8 addrspace(3)* } undef, i8 addrspace(3)* %v191, 0
  %v193 = mul i32 %v184, 16
  %v194 = add i32 %v193, %v24
  %v195 = mul i32 %v187, 64
  %v196 = mul i32 %v186, 64
  %v467 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v468 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v467, i32 0, i32 2
  %v197 = bitcast i32* %v468 to i8*
  %v469 = bitcast i8* %v197 to i32*
  %v198 = load i32, i32* %v469, align 4
  %v470 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v471 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v470, i32 0, i32 3
  %v199 = bitcast i32* %v471 to i8*
  %v472 = bitcast i8* %v199 to i32*
  %v200 = load i32, i32* %v472, align 4
  %v473 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v474 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v473, i32 0, i32 3
  %v201 = bitcast i32* %v474 to i8*
  %v475 = bitcast i8* %v201 to i32*
  %v202 = load i32, i32* %v475, align 4
  %v476 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v477 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v476, i32 0, i32 0
  %v203 = bitcast i32* %v477 to i8*
  %v478 = bitcast i8* %v203 to i32*
  %v204 = load i32, i32* %v478, align 4
  br label %bb2
bb45:
  %v205 = add i32 %v25, 1
  %v206 = insertvalue { i32, i32 } undef, i32 1, 0
  %v207 = insertvalue { i32, i32 } %v206, i32 %v25, 1
  %v208 = extractvalue { i32, i32 } %v207, 0
  %v209 = extractvalue { i32, i32 } %v207, 1
  br label %bb47
bb46:
  %v210 = insertvalue { i32, i32 } undef, i32 0, 0
  %v211 = extractvalue { i32, i32 } %v210, 0
  %v212 = extractvalue { i32, i32 } %v210, 1
  br label %bb47
bb47:
  %v213 = phi i32 [ %v205, %bb45 ], [ %v25, %bb46 ]
  %v214 = phi i32 [ %v208, %bb45 ], [ %v211, %bb46 ]
  %v215 = phi i32 [ %v209, %bb45 ], [ %v212, %bb46 ]
  %v216 = insertvalue { i32, i32 } undef, i32 %v214, 0
  %v217 = insertvalue { i32, i32 } %v216, i32 %v215, 1
  %v218 = extractvalue { i32, i32 } %v217, 0
  %v219 = zext i32 %v218 to i64
  %v220 = icmp eq i64 %v219, 0
  br i1 %v220, label %bb5, label %bb48
bb48:
  %v221 = icmp eq i64 %v219, 1
  br i1 %v221, label %bb4, label %bb3
bb49:
  %v482 = bitcast i8 addrspace(3)* %v81 to float addrspace(3)*
  store float %v77, float addrspace(3)* %v482, align 4
  %v222 = add i32 %v45, 256
  br label %bb9
bb50:
  %v483 = bitcast i8 addrspace(3)* %v118 to float addrspace(3)*
  store float %v114, float addrspace(3)* %v483, align 4
  %v223 = add i32 %v82, 256
  br label %bb16
bb51:
  %v224 = mul i32 %v184, 4
  %v225 = mul i32 %v24, 4
  br label %bb23
bb52:
  %v484 = bitcast i8 addrspace(3)* %v127 to float addrspace(3)*
  %v226 = load float, float addrspace(3)* %v484, align 4
  %v227 = add i32 %v224, 1
  %v228 = mul i32 %v227, 17
  %v229 = add i32 %v228, %v120
  %v230 = zext i32 %v229 to i64
  %v231 = getelementptr i8, i8 addrspace(3)* %v189, i64 0
  %v485 = bitcast i8 addrspace(3)* %v231 to float addrspace(3)*
  %v486 = getelementptr inbounds float, float addrspace(3)* %v485, i64 %v230
  %v232 = bitcast float addrspace(3)* %v486 to i8 addrspace(3)*
  br label %bb53
bb53:
  %v487 = bitcast i8 addrspace(3)* %v232 to float addrspace(3)*
  %v233 = load float, float addrspace(3)* %v487, align 4
  %v234 = add i32 %v224, 2
  %v235 = mul i32 %v234, 17
  %v236 = add i32 %v235, %v120
  %v237 = zext i32 %v236 to i64
  %v238 = getelementptr i8, i8 addrspace(3)* %v189, i64 0
  %v488 = bitcast i8 addrspace(3)* %v238 to float addrspace(3)*
  %v489 = getelementptr inbounds float, float addrspace(3)* %v488, i64 %v237
  %v239 = bitcast float addrspace(3)* %v489 to i8 addrspace(3)*
  br label %bb54
bb54:
  %v490 = bitcast i8 addrspace(3)* %v239 to float addrspace(3)*
  %v240 = load float, float addrspace(3)* %v490, align 4
  %v241 = add i32 %v224, 3
  %v242 = mul i32 %v241, 17
  %v243 = add i32 %v242, %v120
  %v244 = zext i32 %v243 to i64
  %v245 = getelementptr i8, i8 addrspace(3)* %v189, i64 0
  %v491 = bitcast i8 addrspace(3)* %v245 to float addrspace(3)*
  %v492 = getelementptr inbounds float, float addrspace(3)* %v491, i64 %v244
  %v246 = bitcast float addrspace(3)* %v492 to i8 addrspace(3)*
  br label %bb55
bb55:
  %v493 = bitcast i8 addrspace(3)* %v246 to float addrspace(3)*
  %v247 = load float, float addrspace(3)* %v493, align 4
  %v248 = mul i32 %v120, 65
  %v249 = add i32 %v248, %v225
  %v250 = zext i32 %v249 to i64
  %v251 = getelementptr i8, i8 addrspace(3)* %v191, i64 0
  %v494 = bitcast i8 addrspace(3)* %v251 to float addrspace(3)*
  %v495 = getelementptr inbounds float, float addrspace(3)* %v494, i64 %v250
  %v252 = bitcast float addrspace(3)* %v495 to i8 addrspace(3)*
  br label %bb56
bb56:
  %v496 = bitcast i8 addrspace(3)* %v252 to float addrspace(3)*
  %v253 = load float, float addrspace(3)* %v496, align 4
  %v254 = mul i32 %v120, 65
  %v255 = add i32 %v254, %v225
  %v256 = add i32 %v255, 1
  %v257 = zext i32 %v256 to i64
  %v258 = getelementptr i8, i8 addrspace(3)* %v191, i64 0
  %v497 = bitcast i8 addrspace(3)* %v258 to float addrspace(3)*
  %v498 = getelementptr inbounds float, float addrspace(3)* %v497, i64 %v257
  %v259 = bitcast float addrspace(3)* %v498 to i8 addrspace(3)*
  br label %bb57
bb57:
  %v499 = bitcast i8 addrspace(3)* %v259 to float addrspace(3)*
  %v260 = load float, float addrspace(3)* %v499, align 4
  %v261 = mul i32 %v120, 65
  %v262 = add i32 %v261, %v225
  %v263 = add i32 %v262, 2
  %v264 = zext i32 %v263 to i64
  %v265 = getelementptr i8, i8 addrspace(3)* %v191, i64 0
  %v500 = bitcast i8 addrspace(3)* %v265 to float addrspace(3)*
  %v501 = getelementptr inbounds float, float addrspace(3)* %v500, i64 %v264
  %v266 = bitcast float addrspace(3)* %v501 to i8 addrspace(3)*
  br label %bb58
bb58:
  %v502 = bitcast i8 addrspace(3)* %v266 to float addrspace(3)*
  %v267 = load float, float addrspace(3)* %v502, align 4
  %v268 = mul i32 %v120, 65
  %v269 = add i32 %v268, %v225
  %v270 = add i32 %v269, 3
  %v271 = zext i32 %v270 to i64
  %v272 = getelementptr i8, i8 addrspace(3)* %v191, i64 0
  %v503 = bitcast i8 addrspace(3)* %v272 to float addrspace(3)*
  %v504 = getelementptr inbounds float, float addrspace(3)* %v503, i64 %v271
  %v273 = bitcast float addrspace(3)* %v504 to i8 addrspace(3)*
  br label %bb59
bb59:
  %v505 = bitcast i8 addrspace(3)* %v273 to float addrspace(3)*
  %v274 = load float, float addrspace(3)* %v505, align 4
  %v506 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v507 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v506, i32 0, i64 0
  %v275 = bitcast { float, float, float, float }* %v507 to i8*
  %v508 = bitcast i8* %v275 to { float, float, float, float }*
  %v509 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v508, i32 0, i32 0
  %v276 = bitcast float* %v509 to i8*
  %v510 = bitcast i8* %v276 to float*
  %v277 = load float, float* %v510, align 4
  %v511 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v512 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v511, i32 0, i64 0
  %v278 = bitcast { float, float, float, float }* %v512 to i8*
  %v513 = bitcast i8* %v278 to { float, float, float, float }*
  %v514 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v513, i32 0, i32 1
  %v279 = bitcast float* %v514 to i8*
  %v515 = bitcast i8* %v279 to float*
  %v280 = load float, float* %v515, align 4
  %v516 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v517 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v516, i32 0, i64 0
  %v281 = bitcast { float, float, float, float }* %v517 to i8*
  %v518 = bitcast i8* %v281 to { float, float, float, float }*
  %v519 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v518, i32 0, i32 2
  %v282 = bitcast float* %v519 to i8*
  %v520 = bitcast i8* %v282 to float*
  %v283 = load float, float* %v520, align 4
  %v521 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v522 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v521, i32 0, i64 0
  %v284 = bitcast { float, float, float, float }* %v522 to i8*
  %v523 = bitcast i8* %v284 to { float, float, float, float }*
  %v524 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v523, i32 0, i32 3
  %v285 = bitcast float* %v524 to i8*
  %v525 = bitcast i8* %v285 to float*
  %v286 = load float, float* %v525, align 4
  %v287 = call float @__nv_fmaf(float %v253, float %v226, float %v277) #0
  br label %bb60
bb60:
  %v288 = call float @__nv_fmaf(float %v260, float %v226, float %v280) #0
  br label %bb61
bb61:
  %v289 = call float @__nv_fmaf(float %v267, float %v226, float %v283) #0
  br label %bb62
bb62:
  %v290 = call float @__nv_fmaf(float %v274, float %v226, float %v286) #0
  br label %bb63
bb63:
  %v291 = insertvalue { float, float, float, float } undef, float %v287, 0
  %v292 = insertvalue { float, float, float, float } %v291, float %v288, 1
  %v293 = insertvalue { float, float, float, float } %v292, float %v289, 2
  %v294 = insertvalue { float, float, float, float } %v293, float %v290, 3
  %v527 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v528 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v527, i32 0, i64 0
  %v295 = bitcast { float, float, float, float }* %v528 to i8*
  %v529 = bitcast i8* %v295 to { float, float, float, float }*
  store { float, float, float, float } %v294, { float, float, float, float }* %v529, align 4
  %v530 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v531 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v530, i32 0, i64 1
  %v296 = bitcast { float, float, float, float }* %v531 to i8*
  %v532 = bitcast i8* %v296 to { float, float, float, float }*
  %v533 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v532, i32 0, i32 0
  %v297 = bitcast float* %v533 to i8*
  %v534 = bitcast i8* %v297 to float*
  %v298 = load float, float* %v534, align 4
  %v535 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v536 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v535, i32 0, i64 1
  %v299 = bitcast { float, float, float, float }* %v536 to i8*
  %v537 = bitcast i8* %v299 to { float, float, float, float }*
  %v538 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v537, i32 0, i32 1
  %v300 = bitcast float* %v538 to i8*
  %v539 = bitcast i8* %v300 to float*
  %v301 = load float, float* %v539, align 4
  %v540 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v541 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v540, i32 0, i64 1
  %v302 = bitcast { float, float, float, float }* %v541 to i8*
  %v542 = bitcast i8* %v302 to { float, float, float, float }*
  %v543 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v542, i32 0, i32 2
  %v303 = bitcast float* %v543 to i8*
  %v544 = bitcast i8* %v303 to float*
  %v304 = load float, float* %v544, align 4
  %v545 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v546 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v545, i32 0, i64 1
  %v305 = bitcast { float, float, float, float }* %v546 to i8*
  %v547 = bitcast i8* %v305 to { float, float, float, float }*
  %v548 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v547, i32 0, i32 3
  %v306 = bitcast float* %v548 to i8*
  %v549 = bitcast i8* %v306 to float*
  %v307 = load float, float* %v549, align 4
  %v308 = call float @__nv_fmaf(float %v253, float %v233, float %v298) #0
  br label %bb64
bb64:
  %v309 = call float @__nv_fmaf(float %v260, float %v233, float %v301) #0
  br label %bb65
bb65:
  %v310 = call float @__nv_fmaf(float %v267, float %v233, float %v304) #0
  br label %bb66
bb66:
  %v311 = call float @__nv_fmaf(float %v274, float %v233, float %v307) #0
  br label %bb67
bb67:
  %v312 = insertvalue { float, float, float, float } undef, float %v308, 0
  %v313 = insertvalue { float, float, float, float } %v312, float %v309, 1
  %v314 = insertvalue { float, float, float, float } %v313, float %v310, 2
  %v315 = insertvalue { float, float, float, float } %v314, float %v311, 3
  %v551 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v552 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v551, i32 0, i64 1
  %v316 = bitcast { float, float, float, float }* %v552 to i8*
  %v553 = bitcast i8* %v316 to { float, float, float, float }*
  store { float, float, float, float } %v315, { float, float, float, float }* %v553, align 4
  %v554 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v555 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v554, i32 0, i64 2
  %v317 = bitcast { float, float, float, float }* %v555 to i8*
  %v556 = bitcast i8* %v317 to { float, float, float, float }*
  %v557 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v556, i32 0, i32 0
  %v318 = bitcast float* %v557 to i8*
  %v558 = bitcast i8* %v318 to float*
  %v319 = load float, float* %v558, align 4
  %v559 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v560 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v559, i32 0, i64 2
  %v320 = bitcast { float, float, float, float }* %v560 to i8*
  %v561 = bitcast i8* %v320 to { float, float, float, float }*
  %v562 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v561, i32 0, i32 1
  %v321 = bitcast float* %v562 to i8*
  %v563 = bitcast i8* %v321 to float*
  %v322 = load float, float* %v563, align 4
  %v564 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v565 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v564, i32 0, i64 2
  %v323 = bitcast { float, float, float, float }* %v565 to i8*
  %v566 = bitcast i8* %v323 to { float, float, float, float }*
  %v567 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v566, i32 0, i32 2
  %v324 = bitcast float* %v567 to i8*
  %v568 = bitcast i8* %v324 to float*
  %v325 = load float, float* %v568, align 4
  %v569 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v570 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v569, i32 0, i64 2
  %v326 = bitcast { float, float, float, float }* %v570 to i8*
  %v571 = bitcast i8* %v326 to { float, float, float, float }*
  %v572 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v571, i32 0, i32 3
  %v327 = bitcast float* %v572 to i8*
  %v573 = bitcast i8* %v327 to float*
  %v328 = load float, float* %v573, align 4
  %v329 = call float @__nv_fmaf(float %v253, float %v240, float %v319) #0
  br label %bb68
bb68:
  %v330 = call float @__nv_fmaf(float %v260, float %v240, float %v322) #0
  br label %bb69
bb69:
  %v331 = call float @__nv_fmaf(float %v267, float %v240, float %v325) #0
  br label %bb70
bb70:
  %v332 = call float @__nv_fmaf(float %v274, float %v240, float %v328) #0
  br label %bb71
bb71:
  %v333 = insertvalue { float, float, float, float } undef, float %v329, 0
  %v334 = insertvalue { float, float, float, float } %v333, float %v330, 1
  %v335 = insertvalue { float, float, float, float } %v334, float %v331, 2
  %v336 = insertvalue { float, float, float, float } %v335, float %v332, 3
  %v575 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v576 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v575, i32 0, i64 2
  %v337 = bitcast { float, float, float, float }* %v576 to i8*
  %v577 = bitcast i8* %v337 to { float, float, float, float }*
  store { float, float, float, float } %v336, { float, float, float, float }* %v577, align 4
  %v578 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v579 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v578, i32 0, i64 3
  %v338 = bitcast { float, float, float, float }* %v579 to i8*
  %v580 = bitcast i8* %v338 to { float, float, float, float }*
  %v581 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v580, i32 0, i32 0
  %v339 = bitcast float* %v581 to i8*
  %v582 = bitcast i8* %v339 to float*
  %v340 = load float, float* %v582, align 4
  %v583 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v584 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v583, i32 0, i64 3
  %v341 = bitcast { float, float, float, float }* %v584 to i8*
  %v585 = bitcast i8* %v341 to { float, float, float, float }*
  %v586 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v585, i32 0, i32 1
  %v342 = bitcast float* %v586 to i8*
  %v587 = bitcast i8* %v342 to float*
  %v343 = load float, float* %v587, align 4
  %v588 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v589 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v588, i32 0, i64 3
  %v344 = bitcast { float, float, float, float }* %v589 to i8*
  %v590 = bitcast i8* %v344 to { float, float, float, float }*
  %v591 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v590, i32 0, i32 2
  %v345 = bitcast float* %v591 to i8*
  %v592 = bitcast i8* %v345 to float*
  %v346 = load float, float* %v592, align 4
  %v593 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v594 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v593, i32 0, i64 3
  %v347 = bitcast { float, float, float, float }* %v594 to i8*
  %v595 = bitcast i8* %v347 to { float, float, float, float }*
  %v596 = getelementptr inbounds { float, float, float, float }, { float, float, float, float }* %v595, i32 0, i32 3
  %v348 = bitcast float* %v596 to i8*
  %v597 = bitcast i8* %v348 to float*
  %v349 = load float, float* %v597, align 4
  %v350 = call float @__nv_fmaf(float %v253, float %v247, float %v340) #0
  br label %bb72
bb72:
  %v351 = call float @__nv_fmaf(float %v260, float %v247, float %v343) #0
  br label %bb73
bb73:
  %v352 = call float @__nv_fmaf(float %v267, float %v247, float %v346) #0
  br label %bb74
bb74:
  %v353 = call float @__nv_fmaf(float %v274, float %v247, float %v349) #0
  br label %bb75
bb75:
  %v354 = insertvalue { float, float, float, float } undef, float %v350, 0
  %v355 = insertvalue { float, float, float, float } %v354, float %v351, 1
  %v356 = insertvalue { float, float, float, float } %v355, float %v352, 2
  %v357 = insertvalue { float, float, float, float } %v356, float %v353, 3
  %v599 = bitcast i8* %v21 to [4 x { float, float, float, float }]*
  %v600 = getelementptr inbounds [4 x { float, float, float, float }], [4 x { float, float, float, float }]* %v599, i32 0, i64 3
  %v358 = bitcast { float, float, float, float }* %v600 to i8*
  %v601 = bitcast i8* %v358 to { float, float, float, float }*
  store { float, float, float, float } %v357, { float, float, float, float }* %v601, align 4
  %v359 = add i32 %v120, 1
  br label %bb23
bb76:
  %v360 = add i32 %v42, 16
  br label %bb7
bb77:
  call void @llvm.trap() #0
  unreachable
bb78:
  call void @llvm.trap() #0
  unreachable
bb79:
  call void @llvm.trap() #0
  unreachable
}

define void @gemm_naive_cuda_entry_7c653b54ed65ef8f(i8* %v0, i8* %v1, i8* %v2, i8* %v3, i64 %v4, i8* %v5, i64 %v6, i8* %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v10 = insertvalue { i8*, i64 } %v9, i64 %v4, 1
  %v11 = insertvalue { i8*, i64 } undef, i8* %v5, 0
  %v12 = insertvalue { i8*, i64 } %v11, i64 %v6, 1
  %v13 = insertvalue { i8*, i64 } undef, i8* %v7, 0
  %v14 = insertvalue { i8*, i64 } %v13, i64 %v8, 1
  br label %bb0
bb0:
  %v15 = phi i8* [ %v0, %entry ]
  %v16 = phi i8* [ %v1, %entry ]
  %v17 = phi i8* [ %v2, %entry ]
  %v18 = phi { i8*, i64 } [ %v10, %entry ]
  %v19 = phi { i8*, i64 } [ %v12, %entry ]
  %v20 = phi { i8*, i64 } [ %v14, %entry ]
  %v149 = alloca { i32, i32, i32 }, align 4
  %v21 = bitcast { i32, i32, i32 }* %v149 to i8*
  %v22 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v150 = bitcast i8* %v21 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v22, { i32, i32, i32 }* %v150, align 4
  br label %bb1
bb1:
  %v151 = bitcast i8* %v21 to { i32, i32, i32 }*
  %v152 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v151, i32 0, i32 0
  %v23 = bitcast i32* %v152 to i8*
  %v153 = bitcast i8* %v23 to i32*
  %v24 = load i32, i32* %v153, align 4
  %v154 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v155 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v154, i32 0, i32 3
  %v25 = bitcast i32* %v155 to i8*
  %v156 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v156, align 4
  %v27 = icmp uge i32 %v24, %v26
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb3, label %bb2
bb2:
  br label %bb11
bb3:
  %v157 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v158 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v157, i32 0, i32 0
  %v29 = bitcast i32* %v158 to i8*
  %v159 = bitcast i8* %v29 to i32*
  %v30 = load i32, i32* %v159, align 4
  br label %bb4
bb4:
  %v31 = phi i32 [ 0, %bb3 ], [ %v120, %bb10 ]
  %v32 = icmp ult i32 %v31, %v30
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb13, label %bb12
bb5:
  unreachable
bb6:
  %v34 = extractvalue { i32, i32 } %v124, 1
  %v160 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v161 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v160, i32 0, i32 3
  %v35 = bitcast i32* %v161 to i8*
  %v162 = bitcast i8* %v35 to i32*
  %v36 = load i32, i32* %v162, align 4
  br label %bb8
bb7:
  br label %bb11
bb8:
  %v37 = phi float [ 0.0, %bb6 ], [ %v89, %bb9 ]
  %v38 = phi i32 [ 0, %bb6 ], [ %v137, %bb9 ]
  %v39 = icmp ult i32 %v38, %v36
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb17, label %bb16
bb9:
  %v41 = extractvalue { i32, i32 } %v141, 1
  %v163 = bitcast i8* %v21 to { i32, i32, i32 }*
  %v164 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v163, i32 0, i32 2
  %v42 = bitcast i32* %v164 to i8*
  %v165 = bitcast i8* %v42 to i32*
  %v43 = load i32, i32* %v165, align 4
  %v166 = bitcast i8* %v21 to { i32, i32, i32 }*
  %v167 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v166, i32 0, i32 1
  %v44 = bitcast i32* %v167 to i8*
  %v168 = bitcast i8* %v44 to i32*
  %v45 = load i32, i32* %v168, align 4
  %v169 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v170 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v169, i32 0, i32 4
  %v46 = bitcast i32* %v170 to i8*
  %v171 = bitcast i8* %v46 to i32*
  %v47 = load i32, i32* %v171, align 4
  %v48 = mul i32 %v34, %v47
  %v172 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v173 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v172, i32 0, i32 5
  %v49 = bitcast i32* %v173 to i8*
  %v174 = bitcast i8* %v49 to i32*
  %v50 = load i32, i32* %v174, align 4
  %v51 = mul i32 %v43, %v50
  %v52 = add i32 %v48, %v51
  %v175 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v176 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v175, i32 0, i32 6
  %v53 = bitcast i32* %v176 to i8*
  %v177 = bitcast i8* %v53 to i32*
  %v54 = load i32, i32* %v177, align 4
  %v55 = mul i32 %v45, %v54
  %v56 = add i32 %v52, %v55
  %v178 = bitcast i8* %v16 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v179 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v178, i32 0, i32 7
  %v57 = bitcast i32* %v179 to i8*
  %v180 = bitcast i8* %v57 to i32*
  %v58 = load i32, i32* %v180, align 4
  %v59 = mul i32 %v41, %v58
  %v60 = add i32 %v56, %v59
  %v181 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v182 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v181, i32 0, i32 4
  %v61 = bitcast i32* %v182 to i8*
  %v183 = bitcast i8* %v61 to i32*
  %v62 = load i32, i32* %v183, align 4
  %v63 = mul i32 %v34, %v62
  %v184 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v185 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v184, i32 0, i32 5
  %v64 = bitcast i32* %v185 to i8*
  %v186 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v186, align 4
  %v66 = mul i32 %v43, %v65
  %v67 = add i32 %v63, %v66
  %v187 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v188 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v187, i32 0, i32 6
  %v68 = bitcast i32* %v188 to i8*
  %v189 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v189, align 4
  %v70 = mul i32 %v41, %v69
  %v71 = add i32 %v67, %v70
  %v190 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v191 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v190, i32 0, i32 7
  %v72 = bitcast i32* %v191 to i8*
  %v192 = bitcast i8* %v72 to i32*
  %v73 = load i32, i32* %v192, align 4
  %v74 = mul i32 %v24, %v73
  %v75 = add i32 %v71, %v74
  %v76 = zext i32 %v60 to i64
  %v77 = extractvalue { i8*, i64 } %v19, 1
  %v78 = icmp ult i64 %v76, %v77
  %v79 = extractvalue { i8*, i64 } %v19, 0
  %v193 = bitcast i8* %v79 to float*
  %v194 = getelementptr inbounds float, float* %v193, i64 %v76
  %v80 = bitcast float* %v194 to i8*
  %v195 = bitcast i8* %v80 to float*
  %v81 = load float, float* %v195, align 4
  %v82 = zext i32 %v75 to i64
  %v83 = extractvalue { i8*, i64 } %v20, 1
  %v84 = icmp ult i64 %v82, %v83
  %v85 = extractvalue { i8*, i64 } %v20, 0
  %v196 = bitcast i8* %v85 to float*
  %v197 = getelementptr inbounds float, float* %v196, i64 %v82
  %v86 = bitcast float* %v197 to i8*
  %v198 = bitcast i8* %v86 to float*
  %v87 = load float, float* %v198, align 4
  %v88 = fmul contract float %v81, %v87
  %v89 = fadd contract float %v37, %v88
  br label %bb8
bb10:
  %v199 = bitcast i8* %v21 to { i32, i32, i32 }*
  %v200 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v199, i32 0, i32 2
  %v90 = bitcast i32* %v200 to i8*
  %v201 = bitcast i8* %v90 to i32*
  %v91 = load i32, i32* %v201, align 4
  %v202 = bitcast i8* %v21 to { i32, i32, i32 }*
  %v203 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v202, i32 0, i32 1
  %v92 = bitcast i32* %v203 to i8*
  %v204 = bitcast i8* %v92 to i32*
  %v93 = load i32, i32* %v204, align 4
  %v205 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v206 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v205, i32 0, i32 4
  %v94 = bitcast i32* %v206 to i8*
  %v207 = bitcast i8* %v94 to i32*
  %v95 = load i32, i32* %v207, align 4
  %v96 = mul i32 %v34, %v95
  %v208 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v209 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v208, i32 0, i32 5
  %v97 = bitcast i32* %v209 to i8*
  %v210 = bitcast i8* %v97 to i32*
  %v98 = load i32, i32* %v210, align 4
  %v99 = mul i32 %v91, %v98
  %v100 = add i32 %v96, %v99
  %v211 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v212 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v211, i32 0, i32 6
  %v101 = bitcast i32* %v212 to i8*
  %v213 = bitcast i8* %v101 to i32*
  %v102 = load i32, i32* %v213, align 4
  %v103 = mul i32 %v93, %v102
  %v104 = add i32 %v100, %v103
  %v214 = bitcast i8* %v15 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v215 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v214, i32 0, i32 7
  %v105 = bitcast i32* %v215 to i8*
  %v216 = bitcast i8* %v105 to i32*
  %v106 = load i32, i32* %v216, align 4
  %v107 = mul i32 %v24, %v106
  %v108 = add i32 %v104, %v107
  %v109 = zext i32 %v108 to i64
  %v110 = extractvalue { i8*, i64 } %v18, 0
  %v217 = bitcast i8* %v110 to float*
  %v218 = getelementptr inbounds float, float* %v217, i64 %v109
  %v111 = bitcast float* %v218 to i8*
  %v219 = bitcast i8* %v111 to float*
  store float %v37, float* %v219, align 4
  br label %bb4
bb11:
  ret void
bb12:
  %v112 = add i32 %v31, 1
  %v113 = insertvalue { i32, i32 } undef, i32 1, 0
  %v114 = insertvalue { i32, i32 } %v113, i32 %v31, 1
  %v115 = extractvalue { i32, i32 } %v114, 0
  %v116 = extractvalue { i32, i32 } %v114, 1
  br label %bb14
bb13:
  %v117 = insertvalue { i32, i32 } undef, i32 0, 0
  %v118 = extractvalue { i32, i32 } %v117, 0
  %v119 = extractvalue { i32, i32 } %v117, 1
  br label %bb14
bb14:
  %v120 = phi i32 [ %v112, %bb12 ], [ %v31, %bb13 ]
  %v121 = phi i32 [ %v115, %bb12 ], [ %v118, %bb13 ]
  %v122 = phi i32 [ %v116, %bb12 ], [ %v119, %bb13 ]
  %v123 = insertvalue { i32, i32 } undef, i32 %v121, 0
  %v124 = insertvalue { i32, i32 } %v123, i32 %v122, 1
  %v125 = extractvalue { i32, i32 } %v124, 0
  %v126 = zext i32 %v125 to i64
  %v127 = icmp eq i64 %v126, 0
  br i1 %v127, label %bb7, label %bb15
bb15:
  %v128 = icmp eq i64 %v126, 1
  br i1 %v128, label %bb6, label %bb5
bb16:
  %v129 = add i32 %v38, 1
  %v130 = insertvalue { i32, i32 } undef, i32 1, 0
  %v131 = insertvalue { i32, i32 } %v130, i32 %v38, 1
  %v132 = extractvalue { i32, i32 } %v131, 0
  %v133 = extractvalue { i32, i32 } %v131, 1
  br label %bb18
bb17:
  %v134 = insertvalue { i32, i32 } undef, i32 0, 0
  %v135 = extractvalue { i32, i32 } %v134, 0
  %v136 = extractvalue { i32, i32 } %v134, 1
  br label %bb18
bb18:
  %v137 = phi i32 [ %v129, %bb16 ], [ %v38, %bb17 ]
  %v138 = phi i32 [ %v132, %bb16 ], [ %v135, %bb17 ]
  %v139 = phi i32 [ %v133, %bb16 ], [ %v136, %bb17 ]
  %v140 = insertvalue { i32, i32 } undef, i32 %v138, 0
  %v141 = insertvalue { i32, i32 } %v140, i32 %v139, 1
  %v142 = extractvalue { i32, i32 } %v141, 0
  %v143 = zext i32 %v142 to i64
  %v144 = icmp eq i64 %v143, 0
  br i1 %v144, label %bb10, label %bb19
bb19:
  %v145 = icmp eq i64 %v143, 1
  br i1 %v145, label %bb9, label %bb5
}

define void @gpu_ppo_actor_grad_cuda_entry_5af55a16ff51d183(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4, i8* %v5, i64 %v6, i8* %v7, i64 %v8, i8* %v9, i64 %v10, i8* %v11, i64 %v12, i8* %v13, i64 %v14) #0 {
entry:
  %v15 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v16 = insertvalue { i8*, i64 } %v15, i64 %v2, 1
  %v17 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v18 = insertvalue { i8*, i64 } %v17, i64 %v4, 1
  %v19 = insertvalue { i8*, i64 } undef, i8* %v5, 0
  %v20 = insertvalue { i8*, i64 } %v19, i64 %v6, 1
  %v21 = insertvalue { i8*, i64 } undef, i8* %v7, 0
  %v22 = insertvalue { i8*, i64 } %v21, i64 %v8, 1
  %v23 = insertvalue { i8*, i64 } undef, i8* %v9, 0
  %v24 = insertvalue { i8*, i64 } %v23, i64 %v10, 1
  %v25 = insertvalue { i8*, i64 } undef, i8* %v11, 0
  %v26 = insertvalue { i8*, i64 } %v25, i64 %v12, 1
  %v27 = insertvalue { i8*, i64 } undef, i8* %v13, 0
  %v28 = insertvalue { i8*, i64 } %v27, i64 %v14, 1
  br label %bb0
bb0:
  %v29 = phi i8* [ %v0, %entry ]
  %v30 = phi { i8*, i64 } [ %v16, %entry ]
  %v31 = phi { i8*, i64 } [ %v18, %entry ]
  %v32 = phi { i8*, i64 } [ %v20, %entry ]
  %v33 = phi { i8*, i64 } [ %v22, %entry ]
  %v34 = phi { i8*, i64 } [ %v24, %entry ]
  %v35 = phi { i8*, i64 } [ %v26, %entry ]
  %v36 = phi { i8*, i64 } [ %v28, %entry ]
  %v257 = alloca { i32, i32, i32 }, align 4
  %v37 = bitcast { i32, i32, i32 }* %v257 to i8*
  %v258 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v38 = bitcast { { i64, i64 }, i64, i1, [7 x i8] }* %v258 to i8*
  %v39 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v259 = bitcast i8* %v37 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v39, { i32, i32, i32 }* %v259, align 4
  br label %bb1
bb1:
  %v260 = bitcast i8* %v29 to { float, float, float, float, i32, i32, i32, i32 }*
  %v261 = getelementptr inbounds { float, float, float, float, i32, i32, i32, i32 }, { float, float, float, float, i32, i32, i32, i32 }* %v260, i32 0, i32 4
  %v40 = bitcast i32* %v261 to i8*
  %v262 = bitcast i8* %v40 to i32*
  %v41 = load i32, i32* %v262, align 4
  %v42 = zext i32 %v41 to i64
  %v263 = bitcast i8* %v29 to { float, float, float, float, i32, i32, i32, i32 }*
  %v264 = getelementptr inbounds { float, float, float, float, i32, i32, i32, i32 }, { float, float, float, float, i32, i32, i32, i32 }* %v263, i32 0, i32 5
  %v43 = bitcast i32* %v264 to i8*
  %v265 = bitcast i8* %v43 to i32*
  %v44 = load i32, i32* %v265, align 4
  %v45 = zext i32 %v44 to i64
  %v266 = bitcast i8* %v29 to { float, float, float, float, i32, i32, i32, i32 }*
  %v267 = getelementptr inbounds { float, float, float, float, i32, i32, i32, i32 }, { float, float, float, float, i32, i32, i32, i32 }* %v266, i32 0, i32 0
  %v46 = bitcast float* %v267 to i8*
  %v268 = bitcast i8* %v46 to float*
  %v47 = load float, float* %v268, align 4
  %v269 = bitcast i8* %v29 to { float, float, float, float, i32, i32, i32, i32 }*
  %v270 = getelementptr inbounds { float, float, float, float, i32, i32, i32, i32 }, { float, float, float, float, i32, i32, i32, i32 }* %v269, i32 0, i32 2
  %v48 = bitcast float* %v270 to i8*
  %v271 = bitcast i8* %v48 to float*
  %v49 = load float, float* %v271, align 4
  %v272 = bitcast i8* %v29 to { float, float, float, float, i32, i32, i32, i32 }*
  %v273 = getelementptr inbounds { float, float, float, float, i32, i32, i32, i32 }, { float, float, float, float, i32, i32, i32, i32 }* %v272, i32 0, i32 1
  %v50 = bitcast float* %v273 to i8*
  %v274 = bitcast i8* %v50 to float*
  %v51 = load float, float* %v274, align 4
  %v275 = bitcast i8* %v37 to { i32, i32, i32 }*
  %v276 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v275, i32 0, i32 0
  %v52 = bitcast i32* %v276 to i8*
  %v277 = bitcast i8* %v52 to i32*
  %v53 = load i32, i32* %v277, align 4
  %v54 = zext i32 %v53 to i64
  %v55 = insertvalue { i64, i64 } undef, i64 %v54, 0
  %v56 = insertvalue { i64, i64 } %v55, i64 %v45, 1
  %v57 = extractvalue { i64, i64 } %v56, 0
  %v58 = extractvalue { i64, i64 } %v56, 1
  %v59 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCslfDnHtpJyg4_13vortx_shaders(i64 %v57, i64 %v58, i64 16776960) #0
  %v279 = bitcast i8* %v38 to { { i64, i64 }, i64, i1, [7 x i8] }*
  store { { i64, i64 }, i64, i1, [7 x i8] } %v59, { { i64, i64 }, i64, i1, [7 x i8] }* %v279, align 8
  br label %bb23
bb2:
  %v60 = phi i64 [ %v179, %bb19 ], [ %v162, %bb23 ]
  %v61 = phi i64 [ %v180, %bb19 ], [ %v165, %bb23 ]
  %v62 = add i64 %v167, 1
  %v280 = alloca i64, align 8
  %v63 = bitcast i64* %v280 to i8*
  %v281 = bitcast i8* %v63 to i64*
  store i64 %v62, i64* %v281, align 8
  %v282 = bitcast i8* %v63 to { i64 }*
  %v64 = load { i64 }, { i64 }* %v282, align 8
  %v65 = extractvalue { i64 } %v64, 0
  %v66 = sub i64 %v65, 0
  %v67 = icmp ule i64 %v66, 0
  %v68 = add i64 %v66, 0
  %v69 = select i1 %v67, i64 %v68, i64 1
  %v70 = icmp eq i64 %v69, 1
  %v283 = alloca { i64 }, align 8
  %v71 = bitcast { i64 }* %v283 to i8*
  %v284 = bitcast i8* %v71 to { i64 }*
  store { i64 } %v64, { i64 }* %v284, align 8
  %v72 = getelementptr inbounds i8, i8* %v71, i64 0
  %v285 = bitcast i8* %v72 to { { i64 } }*
  %v73 = load { { i64 } }, { { i64 } }* %v285, align 8
  %v286 = alloca { { i64 } }, align 8
  %v74 = bitcast { { i64 } }* %v286 to i8*
  %v287 = bitcast i8* %v74 to { { i64 } }*
  store { { i64 } } %v73, { { i64 } }* %v287, align 8
  %v288 = bitcast i8* %v74 to i64*
  %v75 = load i64, i64* %v288, align 8
  %v76 = icmp ugt i64 %v61, 0
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb25, label %bb24
bb3:
  unreachable
bb4:
  %v78 = extractvalue { i64, i64 } %v184, 1
  br label %bb6
bb5:
  ret void
bb6:
  %v79 = phi float [ 0.0, %bb4 ], [ %v224, %bb32 ]
  %v80 = phi i64 [ 0, %bb4 ], [ %v197, %bb32 ]
  %v81 = icmp ult i64 %v80, %v42
  %v82 = xor i1 %v81, 1
  br i1 %v82, label %bb29, label %bb28
bb7:
  %v83 = extractvalue { i64, i64 } %v201, 1
  %v84 = mul i64 %v83, %v45
  %v85 = add i64 %v84, %v78
  %v86 = extractvalue { i8*, i64 } %v32, 1
  %v87 = icmp ult i64 %v83, %v86
  %v88 = extractvalue { i8*, i64 } %v32, 0
  %v289 = bitcast i8* %v88 to float*
  %v290 = getelementptr inbounds float, float* %v289, i64 %v83
  %v89 = bitcast float* %v290 to i8*
  %v291 = bitcast i8* %v89 to float*
  %v90 = load float, float* %v291, align 4
  %v91 = call float @__nv_expf(float %v90) #0
  br label %bb32
bb8:
  %v92 = extractvalue { i8*, i64 } %v34, 1
  %v93 = icmp ult i64 %v78, %v92
  %v94 = extractvalue { i8*, i64 } %v34, 0
  %v292 = bitcast i8* %v94 to float*
  %v293 = getelementptr inbounds float, float* %v292, i64 %v78
  %v95 = bitcast float* %v293 to i8*
  %v294 = bitcast i8* %v95 to float*
  %v96 = load float, float* %v294, align 4
  %v97 = fsub contract float %v79, %v96
  %v98 = call float @__nv_expf(float %v97) #0
  br label %bb33
bb9:
  %v99 = fadd contract float 1.0, %v47
  %v100 = fcmp ogt float %v98, %v99
  %v101 = xor i1 %v100, 1
  br i1 %v101, label %bb11, label %bb10
bb10:
  br label %bb16
bb11:
  br label %bb12
bb12:
  %v102 = fcmp olt float %v229, 0.0
  %v103 = xor i1 %v102, 1
  br i1 %v103, label %bb14, label %bb13
bb13:
  %v104 = fsub contract float 1.0, %v47
  %v105 = fcmp olt float %v98, %v104
  br label %bb15
bb14:
  br label %bb15
bb15:
  %v106 = phi i1 [ %v105, %bb13 ], [ 0, %bb14 ]
  br label %bb16
bb16:
  %v107 = phi i1 [ 1, %bb10 ], [ %v106, %bb15 ]
  br label %bb17
bb17:
  %v108 = phi i64 [ 0, %bb16 ], [ %v240, %bb22 ]
  %v109 = icmp ult i64 %v108, %v42
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb35, label %bb34
bb18:
  %v111 = extractvalue { i64, i64 } %v244, 1
  %v112 = mul i64 %v111, %v45
  %v113 = add i64 %v112, %v78
  %v114 = extractvalue { i8*, i64 } %v32, 1
  %v115 = icmp ult i64 %v111, %v114
  %v116 = extractvalue { i8*, i64 } %v32, 0
  %v295 = bitcast i8* %v116 to float*
  %v296 = getelementptr inbounds float, float* %v295, i64 %v111
  %v117 = bitcast float* %v296 to i8*
  %v297 = bitcast i8* %v117 to float*
  %v118 = load float, float* %v297, align 4
  %v119 = fmul contract float -2.0, %v118
  %v120 = call float @__nv_expf(float %v119) #0
  br label %bb38
bb19:
  br label %bb2
bb20:
  %v121 = extractvalue { i8*, i64 } %v35, 0
  %v298 = bitcast i8* %v121 to float*
  %v299 = getelementptr inbounds float, float* %v298, i64 %v113
  %v122 = bitcast float* %v299 to i8*
  %v300 = bitcast i8* %v122 to float*
  store float 0.0, float* %v300, align 4
  %v123 = bitcast float %v51 to i32
  %v124 = xor i32 %v123, 2147483648
  %v125 = bitcast i32 %v124 to float
  %v126 = extractvalue { i8*, i64 } %v36, 0
  %v301 = bitcast i8* %v126 to float*
  %v302 = getelementptr inbounds float, float* %v301, i64 %v113
  %v127 = bitcast float* %v302 to i8*
  %v128 = fmul contract float %v125, %v49
  %v303 = bitcast i8* %v127 to float*
  store float %v128, float* %v303, align 4
  br label %bb22
bb21:
  %v129 = extractvalue { i8*, i64 } %v31, 1
  %v130 = icmp ult i64 %v113, %v129
  %v131 = extractvalue { i8*, i64 } %v31, 0
  %v304 = bitcast i8* %v131 to float*
  %v305 = getelementptr inbounds float, float* %v304, i64 %v113
  %v132 = bitcast float* %v305 to i8*
  %v306 = bitcast i8* %v132 to float*
  %v133 = load float, float* %v306, align 4
  %v134 = extractvalue { i8*, i64 } %v30, 1
  %v135 = icmp ult i64 %v113, %v134
  %v136 = extractvalue { i8*, i64 } %v30, 0
  %v307 = bitcast i8* %v136 to float*
  %v308 = getelementptr inbounds float, float* %v307, i64 %v113
  %v137 = bitcast float* %v308 to i8*
  %v309 = bitcast i8* %v137 to float*
  %v138 = load float, float* %v309, align 4
  %v139 = fsub contract float %v133, %v138
  %v140 = fmul contract float %v229, %v98
  %v141 = fmul contract float %v140, %v139
  %v142 = fmul contract float %v141, %v120
  %v143 = bitcast float %v142 to i32
  %v144 = xor i32 %v143, 2147483648
  %v145 = bitcast i32 %v144 to float
  %v146 = extractvalue { i8*, i64 } %v35, 0
  %v310 = bitcast i8* %v146 to float*
  %v311 = getelementptr inbounds float, float* %v310, i64 %v113
  %v147 = bitcast float* %v311 to i8*
  %v148 = fmul contract float %v145, %v49
  %v312 = bitcast i8* %v147 to float*
  store float %v148, float* %v312, align 4
  %v149 = fmul contract float %v139, %v139
  %v150 = fmul contract float %v149, %v120
  %v151 = fsub contract float %v150, 1.0
  %v152 = fmul contract float %v140, %v151
  %v153 = bitcast float %v152 to i32
  %v154 = xor i32 %v153, 2147483648
  %v155 = bitcast i32 %v154 to float
  %v156 = fsub contract float %v155, %v51
  %v157 = extractvalue { i8*, i64 } %v36, 0
  %v313 = bitcast i8* %v157 to float*
  %v314 = getelementptr inbounds float, float* %v313, i64 %v113
  %v158 = bitcast float* %v314 to i8*
  %v159 = fmul contract float %v156, %v49
  %v315 = bitcast i8* %v158 to float*
  store float %v159, float* %v315, align 4
  br label %bb22
bb22:
  br label %bb17
bb23:
  %v316 = bitcast i8* %v38 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v317 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v316, i32 0, i32 0
  %v160 = bitcast { i64, i64 }* %v317 to i8*
  %v318 = bitcast i8* %v160 to { i64, i64 }*
  %v319 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v318, i32 0, i32 0
  %v161 = bitcast i64* %v319 to i8*
  %v320 = bitcast i8* %v161 to i64*
  %v162 = load i64, i64* %v320, align 8
  %v321 = bitcast i8* %v38 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v322 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v321, i32 0, i32 0
  %v163 = bitcast { i64, i64 }* %v322 to i8*
  %v323 = bitcast i8* %v163 to { i64, i64 }*
  %v324 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v323, i32 0, i32 1
  %v164 = bitcast i64* %v324 to i8*
  %v325 = bitcast i8* %v164 to i64*
  %v165 = load i64, i64* %v325, align 8
  %v326 = bitcast i8* %v38 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v327 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v326, i32 0, i32 1
  %v166 = bitcast i64* %v327 to i8*
  %v328 = bitcast i8* %v166 to i64*
  %v167 = load i64, i64* %v328, align 8
  %v329 = bitcast i8* %v38 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v330 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v329, i32 0, i32 2
  %v168 = bitcast i1* %v330 to i8*
  %v331 = bitcast i8* %v168 to i1*
  %v169 = load i1, i1* %v331, align 1
  br label %bb2
bb24:
  %v170 = add i64 %v60, %v75
  %v171 = sub i64 %v61, 1
  %v172 = insertvalue { i64, i64 } undef, i64 1, 0
  %v173 = insertvalue { i64, i64 } %v172, i64 %v60, 1
  %v174 = extractvalue { i64, i64 } %v173, 0
  %v175 = extractvalue { i64, i64 } %v173, 1
  br label %bb26
bb25:
  %v176 = insertvalue { i64, i64 } undef, i64 0, 0
  %v177 = extractvalue { i64, i64 } %v176, 0
  %v178 = extractvalue { i64, i64 } %v176, 1
  br label %bb26
bb26:
  %v179 = phi i64 [ %v170, %bb24 ], [ %v60, %bb25 ]
  %v180 = phi i64 [ %v171, %bb24 ], [ %v61, %bb25 ]
  %v181 = phi i64 [ %v174, %bb24 ], [ %v177, %bb25 ]
  %v182 = phi i64 [ %v175, %bb24 ], [ %v178, %bb25 ]
  %v183 = insertvalue { i64, i64 } undef, i64 %v181, 0
  %v184 = insertvalue { i64, i64 } %v183, i64 %v182, 1
  %v185 = extractvalue { i64, i64 } %v184, 0
  %v186 = bitcast i64 %v185 to i64
  %v187 = icmp eq i64 %v186, 0
  br i1 %v187, label %bb5, label %bb27
bb27:
  %v188 = icmp eq i64 %v186, 1
  br i1 %v188, label %bb4, label %bb3
bb28:
  %v189 = add i64 %v80, 1
  %v190 = insertvalue { i64, i64 } undef, i64 1, 0
  %v191 = insertvalue { i64, i64 } %v190, i64 %v80, 1
  %v192 = extractvalue { i64, i64 } %v191, 0
  %v193 = extractvalue { i64, i64 } %v191, 1
  br label %bb30
bb29:
  %v194 = insertvalue { i64, i64 } undef, i64 0, 0
  %v195 = extractvalue { i64, i64 } %v194, 0
  %v196 = extractvalue { i64, i64 } %v194, 1
  br label %bb30
bb30:
  %v197 = phi i64 [ %v189, %bb28 ], [ %v80, %bb29 ]
  %v198 = phi i64 [ %v192, %bb28 ], [ %v195, %bb29 ]
  %v199 = phi i64 [ %v193, %bb28 ], [ %v196, %bb29 ]
  %v200 = insertvalue { i64, i64 } undef, i64 %v198, 0
  %v201 = insertvalue { i64, i64 } %v200, i64 %v199, 1
  %v202 = extractvalue { i64, i64 } %v201, 0
  %v203 = bitcast i64 %v202 to i64
  %v204 = icmp eq i64 %v203, 0
  br i1 %v204, label %bb8, label %bb31
bb31:
  %v205 = icmp eq i64 %v203, 1
  br i1 %v205, label %bb7, label %bb3
bb32:
  %v206 = extractvalue { i8*, i64 } %v31, 1
  %v207 = icmp ult i64 %v85, %v206
  %v208 = extractvalue { i8*, i64 } %v31, 0
  %v338 = bitcast i8* %v208 to float*
  %v339 = getelementptr inbounds float, float* %v338, i64 %v85
  %v209 = bitcast float* %v339 to i8*
  %v340 = bitcast i8* %v209 to float*
  %v210 = load float, float* %v340, align 4
  %v211 = extractvalue { i8*, i64 } %v30, 1
  %v212 = icmp ult i64 %v85, %v211
  %v213 = extractvalue { i8*, i64 } %v30, 0
  %v341 = bitcast i8* %v213 to float*
  %v342 = getelementptr inbounds float, float* %v341, i64 %v85
  %v214 = bitcast float* %v342 to i8*
  %v343 = bitcast i8* %v214 to float*
  %v215 = load float, float* %v343, align 4
  %v216 = fsub contract float %v210, %v215
  %v217 = fdiv contract float %v216, %v91
  %v218 = fmul contract float -0.5, %v217
  %v219 = fmul contract float %v218, %v217
  %v220 = fsub contract float %v219, %v90
  %v344 = bitcast i8* %v29 to { float, float, float, float, i32, i32, i32, i32 }*
  %v345 = getelementptr inbounds { float, float, float, float, i32, i32, i32, i32 }, { float, float, float, float, i32, i32, i32, i32 }* %v344, i32 0, i32 3
  %v221 = bitcast float* %v345 to i8*
  %v346 = bitcast i8* %v221 to float*
  %v222 = load float, float* %v346, align 4
  %v223 = fsub contract float %v220, %v222
  %v224 = fadd contract float %v79, %v223
  br label %bb6
bb33:
  %v225 = extractvalue { i8*, i64 } %v33, 1
  %v226 = icmp ult i64 %v78, %v225
  %v227 = extractvalue { i8*, i64 } %v33, 0
  %v347 = bitcast i8* %v227 to float*
  %v348 = getelementptr inbounds float, float* %v347, i64 %v78
  %v228 = bitcast float* %v348 to i8*
  %v349 = bitcast i8* %v228 to float*
  %v229 = load float, float* %v349, align 4
  %v230 = fcmp oge float %v229, 0.0
  %v231 = xor i1 %v230, 1
  br i1 %v231, label %bb12, label %bb9
bb34:
  %v232 = add i64 %v108, 1
  %v233 = insertvalue { i64, i64 } undef, i64 1, 0
  %v234 = insertvalue { i64, i64 } %v233, i64 %v108, 1
  %v235 = extractvalue { i64, i64 } %v234, 0
  %v236 = extractvalue { i64, i64 } %v234, 1
  br label %bb36
bb35:
  %v237 = insertvalue { i64, i64 } undef, i64 0, 0
  %v238 = extractvalue { i64, i64 } %v237, 0
  %v239 = extractvalue { i64, i64 } %v237, 1
  br label %bb36
bb36:
  %v240 = phi i64 [ %v232, %bb34 ], [ %v108, %bb35 ]
  %v241 = phi i64 [ %v235, %bb34 ], [ %v238, %bb35 ]
  %v242 = phi i64 [ %v236, %bb34 ], [ %v239, %bb35 ]
  %v243 = insertvalue { i64, i64 } undef, i64 %v241, 0
  %v244 = insertvalue { i64, i64 } %v243, i64 %v242, 1
  %v245 = extractvalue { i64, i64 } %v244, 0
  %v246 = bitcast i64 %v245 to i64
  %v247 = icmp eq i64 %v246, 0
  br i1 %v247, label %bb19, label %bb37
bb37:
  %v248 = icmp eq i64 %v246, 1
  br i1 %v248, label %bb18, label %bb3
bb38:
  %v249 = xor i1 %v107, 1
  br i1 %v249, label %bb21, label %bb20
}

define void @gpu_ppo_value_grad_cuda_entry_71d0ca5bc19893ab(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4, i8* %v5, i64 %v6, i8* %v7, i64 %v8) #0 {
entry:
  %v9 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v10 = insertvalue { i8*, i64 } %v9, i64 %v2, 1
  %v11 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v12 = insertvalue { i8*, i64 } %v11, i64 %v4, 1
  %v13 = insertvalue { i8*, i64 } undef, i8* %v5, 0
  %v14 = insertvalue { i8*, i64 } %v13, i64 %v6, 1
  %v15 = insertvalue { i8*, i64 } undef, i8* %v7, 0
  %v16 = insertvalue { i8*, i64 } %v15, i64 %v8, 1
  br label %bb0
bb0:
  %v17 = phi i8* [ %v0, %entry ]
  %v18 = phi { i8*, i64 } [ %v10, %entry ]
  %v19 = phi { i8*, i64 } [ %v12, %entry ]
  %v20 = phi { i8*, i64 } [ %v14, %entry ]
  %v21 = phi { i8*, i64 } [ %v16, %entry ]
  %v133 = alloca { i32, i32, i32 }, align 4
  %v22 = bitcast { i32, i32, i32 }* %v133 to i8*
  %v134 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v23 = bitcast { { i64, i64 }, i64, i1, [7 x i8] }* %v134 to i8*
  %v24 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v135 = bitcast i8* %v22 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v24, { i32, i32, i32 }* %v135, align 4
  br label %bb1
bb1:
  %v136 = bitcast i8* %v17 to { float, float, float, i32, i32, i32, i32, i32 }*
  %v137 = getelementptr inbounds { float, float, float, i32, i32, i32, i32, i32 }, { float, float, float, i32, i32, i32, i32, i32 }* %v136, i32 0, i32 3
  %v25 = bitcast i32* %v137 to i8*
  %v138 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v138, align 4
  %v27 = zext i32 %v26 to i64
  %v139 = bitcast i8* %v17 to { float, float, float, i32, i32, i32, i32, i32 }*
  %v140 = getelementptr inbounds { float, float, float, i32, i32, i32, i32, i32 }, { float, float, float, i32, i32, i32, i32, i32 }* %v139, i32 0, i32 0
  %v28 = bitcast float* %v140 to i8*
  %v141 = bitcast i8* %v28 to float*
  %v29 = load float, float* %v141, align 4
  %v142 = bitcast i8* %v17 to { float, float, float, i32, i32, i32, i32, i32 }*
  %v143 = getelementptr inbounds { float, float, float, i32, i32, i32, i32, i32 }, { float, float, float, i32, i32, i32, i32, i32 }* %v142, i32 0, i32 2
  %v30 = bitcast float* %v143 to i8*
  %v144 = bitcast i8* %v30 to float*
  %v31 = load float, float* %v144, align 4
  %v145 = bitcast i8* %v22 to { i32, i32, i32 }*
  %v146 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v145, i32 0, i32 0
  %v32 = bitcast i32* %v146 to i8*
  %v147 = bitcast i8* %v32 to i32*
  %v33 = load i32, i32* %v147, align 4
  %v34 = zext i32 %v33 to i64
  %v35 = insertvalue { i64, i64 } undef, i64 %v34, 0
  %v36 = insertvalue { i64, i64 } %v35, i64 %v27, 1
  %v37 = extractvalue { i64, i64 } %v36, 0
  %v38 = extractvalue { i64, i64 } %v36, 1
  %v39 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCslfDnHtpJyg4_13vortx_shaders(i64 %v37, i64 %v38, i64 16776960) #0
  %v149 = bitcast i8* %v23 to { { i64, i64 }, i64, i1, [7 x i8] }*
  store { { i64, i64 }, i64, i1, [7 x i8] } %v39, { { i64, i64 }, i64, i1, [7 x i8] }* %v149, align 8
  br label %bb15
bb2:
  %v40 = phi i64 [ %v119, %bb14 ], [ %v102, %bb15 ]
  %v41 = phi i64 [ %v120, %bb14 ], [ %v105, %bb15 ]
  %v42 = add i64 %v107, 1
  %v150 = alloca i64, align 8
  %v43 = bitcast i64* %v150 to i8*
  %v151 = bitcast i8* %v43 to i64*
  store i64 %v42, i64* %v151, align 8
  %v152 = bitcast i8* %v43 to { i64 }*
  %v44 = load { i64 }, { i64 }* %v152, align 8
  %v45 = extractvalue { i64 } %v44, 0
  %v46 = sub i64 %v45, 0
  %v47 = icmp ule i64 %v46, 0
  %v48 = add i64 %v46, 0
  %v49 = select i1 %v47, i64 %v48, i64 1
  %v50 = icmp eq i64 %v49, 1
  %v153 = alloca { i64 }, align 8
  %v51 = bitcast { i64 }* %v153 to i8*
  %v154 = bitcast i8* %v51 to { i64 }*
  store { i64 } %v44, { i64 }* %v154, align 8
  %v52 = getelementptr inbounds i8, i8* %v51, i64 0
  %v155 = bitcast i8* %v52 to { { i64 } }*
  %v53 = load { { i64 } }, { { i64 } }* %v155, align 8
  %v156 = alloca { { i64 } }, align 8
  %v54 = bitcast { { i64 } }* %v156 to i8*
  %v157 = bitcast i8* %v54 to { { i64 } }*
  store { { i64 } } %v53, { { i64 } }* %v157, align 8
  %v158 = bitcast i8* %v54 to i64*
  %v55 = load i64, i64* %v158, align 8
  %v56 = icmp ugt i64 %v41, 0
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb17, label %bb16
bb3:
  unreachable
bb4:
  %v58 = extractvalue { i64, i64 } %v124, 1
  %v59 = extractvalue { i8*, i64 } %v18, 1
  %v60 = icmp ult i64 %v58, %v59
  %v61 = extractvalue { i8*, i64 } %v18, 0
  %v159 = bitcast i8* %v61 to float*
  %v160 = getelementptr inbounds float, float* %v159, i64 %v58
  %v62 = bitcast float* %v160 to i8*
  %v161 = bitcast i8* %v62 to float*
  %v63 = load float, float* %v161, align 4
  %v64 = extractvalue { i8*, i64 } %v19, 1
  %v65 = icmp ult i64 %v58, %v64
  %v66 = extractvalue { i8*, i64 } %v19, 0
  %v162 = bitcast i8* %v66 to float*
  %v163 = getelementptr inbounds float, float* %v162, i64 %v58
  %v67 = bitcast float* %v163 to i8*
  %v164 = bitcast i8* %v67 to float*
  %v68 = load float, float* %v164, align 4
  %v69 = extractvalue { i8*, i64 } %v20, 1
  %v70 = icmp ult i64 %v58, %v69
  %v71 = extractvalue { i8*, i64 } %v20, 0
  %v165 = bitcast i8* %v71 to float*
  %v166 = getelementptr inbounds float, float* %v165, i64 %v58
  %v72 = bitcast float* %v166 to i8*
  %v167 = bitcast i8* %v72 to float*
  %v73 = load float, float* %v167, align 4
  %v74 = fsub contract float %v63, %v68
  %v75 = fcmp ogt float %v74, %v29
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb7, label %bb6
bb5:
  ret void
bb6:
  br label %bb11
bb7:
  %v77 = bitcast float %v29 to i32
  %v78 = xor i32 %v77, 2147483648
  %v79 = bitcast i32 %v78 to float
  %v80 = fcmp olt float %v74, %v79
  %v81 = xor i1 %v80, 1
  br i1 %v81, label %bb9, label %bb8
bb8:
  br label %bb10
bb9:
  br label %bb10
bb10:
  %v82 = phi float [ %v79, %bb8 ], [ %v74, %bb9 ]
  br label %bb11
bb11:
  %v83 = phi float [ %v29, %bb6 ], [ %v82, %bb10 ]
  %v84 = fadd contract float %v68, %v83
  %v85 = fsub contract float %v63, %v73
  %v86 = fmul contract float %v85, %v85
  %v87 = fsub contract float %v84, %v73
  %v88 = fmul contract float %v87, %v87
  %v89 = fcmp ogt float %v88, %v86
  %v90 = xor i1 %v89, 1
  br i1 %v90, label %bb13, label %bb12
bb12:
  %v91 = fmul contract float 2.0, %v87
  br label %bb14
bb13:
  %v92 = fmul contract float 2.0, %v85
  br label %bb14
bb14:
  %v93 = phi float [ %v91, %bb12 ], [ %v92, %bb13 ]
  %v168 = bitcast i8* %v17 to { float, float, float, i32, i32, i32, i32, i32 }*
  %v169 = getelementptr inbounds { float, float, float, i32, i32, i32, i32, i32 }, { float, float, float, i32, i32, i32, i32, i32 }* %v168, i32 0, i32 1
  %v94 = bitcast float* %v169 to i8*
  %v170 = bitcast i8* %v94 to float*
  %v95 = load float, float* %v170, align 4
  %v96 = fmul contract float %v95, %v93
  %v97 = extractvalue { i8*, i64 } %v21, 0
  %v171 = bitcast i8* %v97 to float*
  %v172 = getelementptr inbounds float, float* %v171, i64 %v58
  %v98 = bitcast float* %v172 to i8*
  %v99 = fmul contract float %v96, %v31
  %v173 = bitcast i8* %v98 to float*
  store float %v99, float* %v173, align 4
  br label %bb2
bb15:
  %v174 = bitcast i8* %v23 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v175 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v174, i32 0, i32 0
  %v100 = bitcast { i64, i64 }* %v175 to i8*
  %v176 = bitcast i8* %v100 to { i64, i64 }*
  %v177 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v176, i32 0, i32 0
  %v101 = bitcast i64* %v177 to i8*
  %v178 = bitcast i8* %v101 to i64*
  %v102 = load i64, i64* %v178, align 8
  %v179 = bitcast i8* %v23 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v180 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v179, i32 0, i32 0
  %v103 = bitcast { i64, i64 }* %v180 to i8*
  %v181 = bitcast i8* %v103 to { i64, i64 }*
  %v182 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v181, i32 0, i32 1
  %v104 = bitcast i64* %v182 to i8*
  %v183 = bitcast i8* %v104 to i64*
  %v105 = load i64, i64* %v183, align 8
  %v184 = bitcast i8* %v23 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v185 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v184, i32 0, i32 1
  %v106 = bitcast i64* %v185 to i8*
  %v186 = bitcast i8* %v106 to i64*
  %v107 = load i64, i64* %v186, align 8
  %v187 = bitcast i8* %v23 to { { i64, i64 }, i64, i1, [7 x i8] }*
  %v188 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, { { i64, i64 }, i64, i1, [7 x i8] }* %v187, i32 0, i32 2
  %v108 = bitcast i1* %v188 to i8*
  %v189 = bitcast i8* %v108 to i1*
  %v109 = load i1, i1* %v189, align 1
  br label %bb2
bb16:
  %v110 = add i64 %v40, %v55
  %v111 = sub i64 %v41, 1
  %v112 = insertvalue { i64, i64 } undef, i64 1, 0
  %v113 = insertvalue { i64, i64 } %v112, i64 %v40, 1
  %v114 = extractvalue { i64, i64 } %v113, 0
  %v115 = extractvalue { i64, i64 } %v113, 1
  br label %bb18
bb17:
  %v116 = insertvalue { i64, i64 } undef, i64 0, 0
  %v117 = extractvalue { i64, i64 } %v116, 0
  %v118 = extractvalue { i64, i64 } %v116, 1
  br label %bb18
bb18:
  %v119 = phi i64 [ %v110, %bb16 ], [ %v40, %bb17 ]
  %v120 = phi i64 [ %v111, %bb16 ], [ %v41, %bb17 ]
  %v121 = phi i64 [ %v114, %bb16 ], [ %v117, %bb17 ]
  %v122 = phi i64 [ %v115, %bb16 ], [ %v118, %bb17 ]
  %v123 = insertvalue { i64, i64 } undef, i64 %v121, 0
  %v124 = insertvalue { i64, i64 } %v123, i64 %v122, 1
  %v125 = extractvalue { i64, i64 } %v124, 0
  %v126 = bitcast i64 %v125 to i64
  %v127 = icmp eq i64 %v126, 0
  br i1 %v127, label %bb5, label %bb19
bb19:
  %v128 = icmp eq i64 %v126, 1
  br i1 %v128, label %bb4, label %bb3
}

define void @contiguous_with_offset_cuda_entry_730898f1e627395f(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4, i8* %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v2, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v4, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi { i8*, i64 } [ %v7, %entry ]
  %v12 = phi { i8*, i64 } [ %v9, %entry ]
  %v13 = phi i8* [ %v5, %entry ]
  %v14 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  br label %bb1
bb1:
  %v26 = bitcast i8* %v13 to i32*
  %v15 = load i32, i32* %v26, align 4
  %v16 = extractvalue { i32, i32, i32 } %v14, 0
  %v17 = extractvalue { i32, i32, i32 } %v14, 1
  %v18 = extractvalue { i32, i32, i32 } %v14, 2
  %v19 = extractvalue { i8*, i64 } %v11, 0
  %v20 = extractvalue { i8*, i64 } %v11, 1
  %v21 = extractvalue { i8*, i64 } %v12, 0
  %v22 = extractvalue { i8*, i64 } %v12, 1
  call void @vortx_shaders__linalg__contiguous__contiguous_impl(i32 %v16, i32 %v17, i32 %v18, i8* %v10, i8* %v19, i64 %v20, i8* %v21, i64 %v22, i32 %v15) #0
  br label %bb2
bb2:
  ret void
}

define void @contiguous_cuda_entry_6c8a092756f20c36(i8* %v0, i8* %v1, i64 %v2, i8* %v3, i64 %v4) #0 {
entry:
  %v5 = insertvalue { i8*, i64 } undef, i8* %v1, 0
  %v6 = insertvalue { i8*, i64 } %v5, i64 %v2, 1
  %v7 = insertvalue { i8*, i64 } undef, i8* %v3, 0
  %v8 = insertvalue { i8*, i64 } %v7, i64 %v4, 1
  br label %bb0
bb0:
  %v9 = phi i8* [ %v0, %entry ]
  %v10 = phi { i8*, i64 } [ %v6, %entry ]
  %v11 = phi { i8*, i64 } [ %v8, %entry ]
  %v12 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  br label %bb1
bb1:
  %v13 = extractvalue { i32, i32, i32 } %v12, 0
  %v14 = extractvalue { i32, i32, i32 } %v12, 1
  %v15 = extractvalue { i32, i32, i32 } %v12, 2
  %v16 = extractvalue { i8*, i64 } %v10, 0
  %v17 = extractvalue { i8*, i64 } %v10, 1
  %v18 = extractvalue { i8*, i64 } %v11, 0
  %v19 = extractvalue { i8*, i64 } %v11, 1
  call void @vortx_shaders__linalg__contiguous__contiguous_impl(i32 %v13, i32 %v14, i32 %v15, i8* %v9, i8* %v16, i64 %v17, i8* %v18, i64 %v19, i32 0) #0
  br label %bb2
bb2:
  ret void
}

define void @repeat_cuda_entry_136129c3b46edf0e(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5) #0 {
entry:
  %v6 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v7 = insertvalue { i8*, i64 } %v6, i64 %v3, 1
  %v8 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v9 = insertvalue { i8*, i64 } %v8, i64 %v5, 1
  br label %bb0
bb0:
  %v10 = phi i8* [ %v0, %entry ]
  %v11 = phi i8* [ %v1, %entry ]
  %v12 = phi { i8*, i64 } [ %v7, %entry ]
  %v13 = phi { i8*, i64 } [ %v9, %entry ]
  %v113 = alloca { i32, i32, i32 }, align 4
  %v14 = bitcast { i32, i32, i32 }* %v113 to i8*
  %v114 = alloca { i32, i32, i32, i32 }, align 4
  %v15 = bitcast { i32, i32, i32, i32 }* %v114 to i8*
  %v115 = alloca { i32, i32, i32, i32 }, align 4
  %v16 = bitcast { i32, i32, i32, i32 }* %v115 to i8*
  %v17 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v116 = bitcast i8* %v14 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v17, { i32, i32, i32 }* %v116, align 4
  br label %bb1
bb1:
  %v117 = bitcast i8* %v14 to { i32, i32, i32 }*
  %v118 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v117, i32 0, i32 0
  %v18 = bitcast i32* %v118 to i8*
  %v119 = bitcast i8* %v18 to i32*
  %v19 = load i32, i32* %v119, align 4
  %v120 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v121 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v120, i32 0, i32 0
  %v20 = bitcast i32* %v121 to i8*
  %v122 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v122, align 4
  %v123 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v124 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v123, i32 0, i32 1
  %v22 = bitcast i32* %v124 to i8*
  %v125 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v125, align 4
  %v24 = mul i32 %v21, %v23
  %v126 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v127 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v126, i32 0, i32 2
  %v25 = bitcast i32* %v127 to i8*
  %v128 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v128, align 4
  %v27 = mul i32 %v24, %v26
  %v129 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v130 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v129, i32 0, i32 3
  %v28 = bitcast i32* %v130 to i8*
  %v131 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v131, align 4
  %v30 = mul i32 %v27, %v29
  %v31 = icmp uge i32 %v19, %v30
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb3, label %bb2
bb2:
  br label %bb5
bb3:
  %v33 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v10, i32 %v19) #0
  %v132 = bitcast i8* %v15 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v33, { i32, i32, i32, i32 }* %v132, align 4
  br label %bb4
bb4:
  %v133 = bitcast i8* %v15 to { i32, i32, i32, i32 }*
  %v134 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v133, i32 0, i32 0
  %v34 = bitcast i32* %v134 to i8*
  %v135 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v135, align 4
  %v136 = bitcast i8* %v15 to { i32, i32, i32, i32 }*
  %v137 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v136, i32 0, i32 1
  %v36 = bitcast i32* %v137 to i8*
  %v138 = bitcast i8* %v36 to i32*
  %v37 = load i32, i32* %v138, align 4
  %v139 = bitcast i8* %v15 to { i32, i32, i32, i32 }*
  %v140 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v139, i32 0, i32 2
  %v38 = bitcast i32* %v140 to i8*
  %v141 = bitcast i8* %v38 to i32*
  %v39 = load i32, i32* %v141, align 4
  %v142 = bitcast i8* %v15 to { i32, i32, i32, i32 }*
  %v143 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v142, i32 0, i32 3
  %v40 = bitcast i32* %v143 to i8*
  %v144 = bitcast i8* %v40 to i32*
  %v41 = load i32, i32* %v144, align 4
  %v145 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v146 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v145, i32 0, i32 4
  %v42 = bitcast i32* %v146 to i8*
  %v147 = bitcast i8* %v42 to i32*
  %v43 = load i32, i32* %v147, align 4
  %v44 = mul i32 %v35, %v43
  %v148 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v149 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v148, i32 0, i32 5
  %v45 = bitcast i32* %v149 to i8*
  %v150 = bitcast i8* %v45 to i32*
  %v46 = load i32, i32* %v150, align 4
  %v47 = mul i32 %v37, %v46
  %v48 = add i32 %v44, %v47
  %v151 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v152 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v151, i32 0, i32 6
  %v49 = bitcast i32* %v152 to i8*
  %v153 = bitcast i8* %v49 to i32*
  %v50 = load i32, i32* %v153, align 4
  %v51 = mul i32 %v39, %v50
  %v52 = add i32 %v48, %v51
  %v154 = bitcast i8* %v10 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v155 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v154, i32 0, i32 7
  %v53 = bitcast i32* %v155 to i8*
  %v156 = bitcast i8* %v53 to i32*
  %v54 = load i32, i32* %v156, align 4
  %v55 = mul i32 %v41, %v54
  %v56 = add i32 %v52, %v55
  %v57 = zext i32 %v56 to i64
  %v157 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v158 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v157, i32 0, i32 0
  %v58 = bitcast i32* %v158 to i8*
  %v159 = bitcast i8* %v58 to i32*
  %v59 = load i32, i32* %v159, align 4
  %v160 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v161 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v160, i32 0, i32 1
  %v60 = bitcast i32* %v161 to i8*
  %v162 = bitcast i8* %v60 to i32*
  %v61 = load i32, i32* %v162, align 4
  %v163 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v164 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v163, i32 0, i32 2
  %v62 = bitcast i32* %v164 to i8*
  %v165 = bitcast i8* %v62 to i32*
  %v63 = load i32, i32* %v165, align 4
  %v166 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v167 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v166, i32 0, i32 3
  %v64 = bitcast i32* %v167 to i8*
  %v168 = bitcast i8* %v64 to i32*
  %v65 = load i32, i32* %v168, align 4
  %v66 = insertvalue { i32, i32, i32, i32 } undef, i32 %v59, 0
  %v67 = insertvalue { i32, i32, i32, i32 } %v66, i32 %v61, 1
  %v68 = insertvalue { i32, i32, i32, i32 } %v67, i32 %v63, 2
  %v69 = insertvalue { i32, i32, i32, i32 } %v68, i32 %v65, 3
  %v170 = bitcast i8* %v15 to { i32, i32, i32, i32 }*
  %v70 = load { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v170, align 4
  %v71 = extractvalue { i32, i32, i32, i32 } %v70, 0
  %v72 = extractvalue { i32, i32, i32, i32 } %v70, 1
  %v73 = extractvalue { i32, i32, i32, i32 } %v70, 2
  %v74 = extractvalue { i32, i32, i32, i32 } %v70, 3
  %v75 = extractvalue { i32, i32, i32, i32 } %v69, 0
  %v76 = extractvalue { i32, i32, i32, i32 } %v69, 1
  %v77 = extractvalue { i32, i32, i32, i32 } %v69, 2
  %v78 = extractvalue { i32, i32, i32, i32 } %v69, 3
  %v79 = call { i32, i32, i32, i32 } @_glamx__UVec4_as_std__ops__Rem___rem(i32 %v71, i32 %v72, i32 %v73, i32 %v74, i32 %v75, i32 %v76, i32 %v77, i32 %v78) #0
  %v171 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v79, { i32, i32, i32, i32 }* %v171, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v172 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v173 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v172, i32 0, i32 0
  %v80 = bitcast i32* %v173 to i8*
  %v174 = bitcast i8* %v80 to i32*
  %v81 = load i32, i32* %v174, align 4
  %v175 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v176 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v175, i32 0, i32 4
  %v82 = bitcast i32* %v176 to i8*
  %v177 = bitcast i8* %v82 to i32*
  %v83 = load i32, i32* %v177, align 4
  %v84 = mul i32 %v81, %v83
  %v178 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v179 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v178, i32 0, i32 1
  %v85 = bitcast i32* %v179 to i8*
  %v180 = bitcast i8* %v85 to i32*
  %v86 = load i32, i32* %v180, align 4
  %v181 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v182 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v181, i32 0, i32 5
  %v87 = bitcast i32* %v182 to i8*
  %v183 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v183, align 4
  %v89 = mul i32 %v86, %v88
  %v90 = add i32 %v84, %v89
  %v184 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v185 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v184, i32 0, i32 2
  %v91 = bitcast i32* %v185 to i8*
  %v186 = bitcast i8* %v91 to i32*
  %v92 = load i32, i32* %v186, align 4
  %v187 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v188 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v187, i32 0, i32 6
  %v93 = bitcast i32* %v188 to i8*
  %v189 = bitcast i8* %v93 to i32*
  %v94 = load i32, i32* %v189, align 4
  %v95 = mul i32 %v92, %v94
  %v96 = add i32 %v90, %v95
  %v190 = bitcast i8* %v16 to { i32, i32, i32, i32 }*
  %v191 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v190, i32 0, i32 3
  %v97 = bitcast i32* %v191 to i8*
  %v192 = bitcast i8* %v97 to i32*
  %v98 = load i32, i32* %v192, align 4
  %v193 = bitcast i8* %v11 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v194 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v193, i32 0, i32 7
  %v99 = bitcast i32* %v194 to i8*
  %v195 = bitcast i8* %v99 to i32*
  %v100 = load i32, i32* %v195, align 4
  %v101 = mul i32 %v98, %v100
  %v102 = add i32 %v96, %v101
  %v103 = zext i32 %v102 to i64
  %v104 = extractvalue { i8*, i64 } %v13, 1
  %v105 = icmp ult i64 %v103, %v104
  %v106 = extractvalue { i8*, i64 } %v13, 0
  %v196 = bitcast i8* %v106 to float*
  %v197 = getelementptr inbounds float, float* %v196, i64 %v103
  %v107 = bitcast float* %v197 to i8*
  %v198 = bitcast i8* %v107 to float*
  %v108 = load float, float* %v198, align 4
  %v109 = extractvalue { i8*, i64 } %v12, 0
  %v199 = bitcast i8* %v109 to float*
  %v200 = getelementptr inbounds float, float* %v199, i64 %v57
  %v110 = bitcast float* %v200 to i8*
  %v201 = bitcast i8* %v110 to float*
  store float %v108, float* %v201, align 4
  br label %bb5
}

declare float @__nv_sqrtf(float)

define void @gpu_adam_cuda_entry_1d28efac816be4b8(i8* %v0, i8* %v1, i8* %v2, i64 %v3, i8* %v4, i64 %v5, i8* %v6, i64 %v7, i8* %v8, i64 %v9) #0 {
entry:
  %v10 = insertvalue { i8*, i64 } undef, i8* %v2, 0
  %v11 = insertvalue { i8*, i64 } %v10, i64 %v3, 1
  %v12 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v13 = insertvalue { i8*, i64 } %v12, i64 %v5, 1
  %v14 = insertvalue { i8*, i64 } undef, i8* %v6, 0
  %v15 = insertvalue { i8*, i64 } %v14, i64 %v7, 1
  %v16 = insertvalue { i8*, i64 } undef, i8* %v8, 0
  %v17 = insertvalue { i8*, i64 } %v16, i64 %v9, 1
  br label %bb0
bb0:
  %v18 = phi i8* [ %v0, %entry ]
  %v19 = phi i8* [ %v1, %entry ]
  %v20 = phi { i8*, i64 } [ %v11, %entry ]
  %v21 = phi { i8*, i64 } [ %v13, %entry ]
  %v22 = phi { i8*, i64 } [ %v15, %entry ]
  %v23 = phi { i8*, i64 } [ %v17, %entry ]
  %v173 = alloca { i32, i32, i32 }, align 4
  %v24 = bitcast { i32, i32, i32 }* %v173 to i8*
  %v174 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v25 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v174 to i8*
  %v175 = alloca { i32, i32, i32, i32 }, align 4
  %v26 = bitcast { i32, i32, i32, i32 }* %v175 to i8*
  %v27 = call { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() #0
  %v176 = bitcast i8* %v24 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v27, { i32, i32, i32 }* %v176, align 4
  br label %bb1
bb1:
  %v177 = bitcast i8* %v24 to { i32, i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v177, i32 0, i32 0
  %v28 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v28 to i32*
  %v29 = load i32, i32* %v179, align 4
  %v180 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v181 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v180, i32 0, i32 0
  %v30 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v30 to i32*
  %v31 = load i32, i32* %v182, align 4
  %v183 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v184 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v183, i32 0, i32 1
  %v32 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v32 to i32*
  %v33 = load i32, i32* %v185, align 4
  %v34 = mul i32 %v31, %v33
  %v186 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v187 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v186, i32 0, i32 2
  %v35 = bitcast i32* %v187 to i8*
  %v188 = bitcast i8* %v35 to i32*
  %v36 = load i32, i32* %v188, align 4
  %v37 = mul i32 %v34, %v36
  %v189 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v190 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v189, i32 0, i32 3
  %v38 = bitcast i32* %v190 to i8*
  %v191 = bitcast i8* %v38 to i32*
  %v39 = load i32, i32* %v191, align 4
  %v40 = mul i32 %v37, %v39
  %v41 = insertvalue { i32, i32 } undef, i32 %v29, 0
  %v42 = insertvalue { i32, i32 } %v41, i32 %v40, 1
  %v43 = extractvalue { i32, i32 } %v42, 0
  %v44 = extractvalue { i32, i32 } %v42, 1
  %v45 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v43, i32 %v44, i64 16776960) #0
  %v193 = bitcast i8* %v25 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v45, { { i32, i32 }, i64, i1, [7 x i8] }* %v193, align 8
  br label %bb7
bb2:
  %v46 = phi i32 [ %v130, %bb7 ], [ %v147, %bb15 ]
  %v47 = phi i32 [ %v133, %bb7 ], [ %v148, %bb15 ]
  %v48 = add i64 %v135, 1
  %v194 = alloca i64, align 8
  %v49 = bitcast i64* %v194 to i8*
  %v195 = bitcast i8* %v49 to i64*
  store i64 %v48, i64* %v195, align 8
  %v196 = bitcast i8* %v49 to { i64 }*
  %v50 = load { i64 }, { i64 }* %v196, align 8
  %v51 = extractvalue { i64 } %v50, 0
  %v52 = sub i64 %v51, 0
  %v53 = icmp ule i64 %v52, 0
  %v54 = add i64 %v52, 0
  %v55 = select i1 %v53, i64 %v54, i64 1
  %v56 = icmp eq i64 %v55, 1
  %v197 = alloca { i64 }, align 8
  %v57 = bitcast { i64 }* %v197 to i8*
  %v198 = bitcast i8* %v57 to { i64 }*
  store { i64 } %v50, { i64 }* %v198, align 8
  %v58 = getelementptr inbounds i8, i8* %v57, i64 0
  %v199 = bitcast i8* %v58 to { { i64 } }*
  %v59 = load { { i64 } }, { { i64 } }* %v199, align 8
  %v200 = alloca { { i64 } }, align 8
  %v60 = bitcast { { i64 } }* %v200 to i8*
  %v201 = bitcast i8* %v60 to { { i64 } }*
  store { { i64 } } %v59, { { i64 } }* %v201, align 8
  %v202 = bitcast i8* %v60 to i64*
  %v61 = load i64, i64* %v202, align 8
  %v62 = icmp ugt i64 %v61, 4294967295
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb13, label %bb12
bb3:
  unreachable
bb4:
  %v64 = extractvalue { i32, i32 } %v152, 1
  %v65 = call { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v18, i32 %v64) #0
  %v203 = bitcast i8* %v26 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v65, { i32, i32, i32, i32 }* %v203, align 4
  br label %bb6
bb5:
  ret void
bb6:
  %v204 = bitcast i8* %v26 to { i32, i32, i32, i32 }*
  %v205 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v204, i32 0, i32 0
  %v66 = bitcast i32* %v205 to i8*
  %v206 = bitcast i8* %v66 to i32*
  %v67 = load i32, i32* %v206, align 4
  %v207 = bitcast i8* %v26 to { i32, i32, i32, i32 }*
  %v208 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v207, i32 0, i32 1
  %v68 = bitcast i32* %v208 to i8*
  %v209 = bitcast i8* %v68 to i32*
  %v69 = load i32, i32* %v209, align 4
  %v210 = bitcast i8* %v26 to { i32, i32, i32, i32 }*
  %v211 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v210, i32 0, i32 2
  %v70 = bitcast i32* %v211 to i8*
  %v212 = bitcast i8* %v70 to i32*
  %v71 = load i32, i32* %v212, align 4
  %v213 = bitcast i8* %v26 to { i32, i32, i32, i32 }*
  %v214 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v213, i32 0, i32 3
  %v72 = bitcast i32* %v214 to i8*
  %v215 = bitcast i8* %v72 to i32*
  %v73 = load i32, i32* %v215, align 4
  %v216 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v217 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v216, i32 0, i32 4
  %v74 = bitcast i32* %v217 to i8*
  %v218 = bitcast i8* %v74 to i32*
  %v75 = load i32, i32* %v218, align 4
  %v76 = mul i32 %v67, %v75
  %v219 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v220 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v219, i32 0, i32 5
  %v77 = bitcast i32* %v220 to i8*
  %v221 = bitcast i8* %v77 to i32*
  %v78 = load i32, i32* %v221, align 4
  %v79 = mul i32 %v69, %v78
  %v80 = add i32 %v76, %v79
  %v222 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v223 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v222, i32 0, i32 6
  %v81 = bitcast i32* %v223 to i8*
  %v224 = bitcast i8* %v81 to i32*
  %v82 = load i32, i32* %v224, align 4
  %v83 = mul i32 %v71, %v82
  %v84 = add i32 %v80, %v83
  %v225 = bitcast i8* %v18 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v226 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v225, i32 0, i32 7
  %v85 = bitcast i32* %v226 to i8*
  %v227 = bitcast i8* %v85 to i32*
  %v86 = load i32, i32* %v227, align 4
  %v87 = mul i32 %v73, %v86
  %v88 = add i32 %v84, %v87
  %v89 = zext i32 %v88 to i64
  %v90 = extractvalue { i8*, i64 } %v21, 1
  %v91 = icmp ult i64 %v89, %v90
  %v92 = extractvalue { i8*, i64 } %v21, 0
  %v228 = bitcast i8* %v92 to float*
  %v229 = getelementptr inbounds float, float* %v228, i64 %v89
  %v93 = bitcast float* %v229 to i8*
  %v230 = bitcast i8* %v93 to float*
  %v94 = load float, float* %v230, align 4
  %v95 = extractvalue { i8*, i64 } %v22, 0
  %v231 = bitcast i8* %v95 to float*
  %v232 = getelementptr inbounds float, float* %v231, i64 %v89
  %v96 = bitcast float* %v232 to i8*
  %v233 = bitcast i8* %v96 to float*
  %v97 = load float, float* %v233, align 4
  %v98 = extractvalue { i8*, i64 } %v23, 0
  %v234 = bitcast i8* %v98 to float*
  %v235 = getelementptr inbounds float, float* %v234, i64 %v89
  %v99 = bitcast float* %v235 to i8*
  %v236 = bitcast i8* %v99 to float*
  %v100 = load float, float* %v236, align 4
  %v237 = bitcast i8* %v19 to { float, float, float, float, float, float, float, float }*
  %v238 = getelementptr inbounds { float, float, float, float, float, float, float, float }, { float, float, float, float, float, float, float, float }* %v237, i32 0, i32 1
  %v101 = bitcast float* %v238 to i8*
  %v239 = bitcast i8* %v101 to float*
  %v102 = load float, float* %v239, align 4
  %v103 = fmul contract float %v102, %v97
  %v104 = fsub contract float 1.0, %v102
  %v105 = fmul contract float %v104, %v94
  %v106 = fadd contract float %v103, %v105
  %v240 = bitcast i8* %v19 to { float, float, float, float, float, float, float, float }*
  %v241 = getelementptr inbounds { float, float, float, float, float, float, float, float }, { float, float, float, float, float, float, float, float }* %v240, i32 0, i32 2
  %v107 = bitcast float* %v241 to i8*
  %v242 = bitcast i8* %v107 to float*
  %v108 = load float, float* %v242, align 4
  %v109 = fmul contract float %v108, %v100
  %v110 = fsub contract float 1.0, %v108
  %v111 = fmul contract float %v110, %v94
  %v112 = fmul contract float %v111, %v94
  %v113 = fadd contract float %v109, %v112
  %v114 = extractvalue { i8*, i64 } %v22, 0
  %v243 = bitcast i8* %v114 to float*
  %v244 = getelementptr inbounds float, float* %v243, i64 %v89
  %v115 = bitcast float* %v244 to i8*
  %v245 = bitcast i8* %v115 to float*
  store float %v106, float* %v245, align 4
  %v116 = extractvalue { i8*, i64 } %v23, 0
  %v246 = bitcast i8* %v116 to float*
  %v247 = getelementptr inbounds float, float* %v246, i64 %v89
  %v117 = bitcast float* %v247 to i8*
  %v248 = bitcast i8* %v117 to float*
  store float %v113, float* %v248, align 4
  %v249 = bitcast i8* %v19 to { float, float, float, float, float, float, float, float }*
  %v250 = getelementptr inbounds { float, float, float, float, float, float, float, float }, { float, float, float, float, float, float, float, float }* %v249, i32 0, i32 4
  %v118 = bitcast float* %v250 to i8*
  %v251 = bitcast i8* %v118 to float*
  %v119 = load float, float* %v251, align 4
  %v120 = fdiv contract float %v106, %v119
  %v252 = bitcast i8* %v19 to { float, float, float, float, float, float, float, float }*
  %v253 = getelementptr inbounds { float, float, float, float, float, float, float, float }, { float, float, float, float, float, float, float, float }* %v252, i32 0, i32 5
  %v121 = bitcast float* %v253 to i8*
  %v254 = bitcast i8* %v121 to float*
  %v122 = load float, float* %v254, align 4
  %v123 = fdiv contract float %v113, %v122
  %v255 = bitcast i8* %v19 to { float, float, float, float, float, float, float, float }*
  %v256 = getelementptr inbounds { float, float, float, float, float, float, float, float }, { float, float, float, float, float, float, float, float }* %v255, i32 0, i32 0
  %v124 = bitcast float* %v256 to i8*
  %v257 = bitcast i8* %v124 to float*
  %v125 = load float, float* %v257, align 4
  %v126 = fmul contract float %v125, %v120
  %v127 = call float @__nv_sqrtf(float %v123) #0
  br label %bb15
bb7:
  %v258 = bitcast i8* %v25 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v259 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v258, i32 0, i32 0
  %v128 = bitcast { i32, i32 }* %v259 to i8*
  %v260 = bitcast i8* %v128 to { i32, i32 }*
  %v261 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v260, i32 0, i32 0
  %v129 = bitcast i32* %v261 to i8*
  %v262 = bitcast i8* %v129 to i32*
  %v130 = load i32, i32* %v262, align 4
  %v263 = bitcast i8* %v25 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v264 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v263, i32 0, i32 0
  %v131 = bitcast { i32, i32 }* %v264 to i8*
  %v265 = bitcast i8* %v131 to { i32, i32 }*
  %v266 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v265, i32 0, i32 1
  %v132 = bitcast i32* %v266 to i8*
  %v267 = bitcast i8* %v132 to i32*
  %v133 = load i32, i32* %v267, align 4
  %v268 = bitcast i8* %v25 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v269 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v268, i32 0, i32 1
  %v134 = bitcast i64* %v269 to i8*
  %v270 = bitcast i8* %v134 to i64*
  %v135 = load i64, i64* %v270, align 8
  %v271 = bitcast i8* %v25 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v272 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v271, i32 0, i32 2
  %v136 = bitcast i1* %v272 to i8*
  %v273 = bitcast i8* %v136 to i1*
  %v137 = load i1, i1* %v273, align 1
  br label %bb2
bb8:
  %v138 = add i32 %v46, %v158
  %v139 = sub i32 %v47, 1
  %v140 = insertvalue { i32, i32 } undef, i32 1, 0
  %v141 = insertvalue { i32, i32 } %v140, i32 %v46, 1
  %v142 = extractvalue { i32, i32 } %v141, 0
  %v143 = extractvalue { i32, i32 } %v141, 1
  br label %bb10
bb9:
  %v144 = insertvalue { i32, i32 } undef, i32 0, 0
  %v145 = extractvalue { i32, i32 } %v144, 0
  %v146 = extractvalue { i32, i32 } %v144, 1
  br label %bb10
bb10:
  %v147 = phi i32 [ %v138, %bb8 ], [ %v46, %bb9 ]
  %v148 = phi i32 [ %v139, %bb8 ], [ %v47, %bb9 ]
  %v149 = phi i32 [ %v142, %bb8 ], [ %v145, %bb9 ]
  %v150 = phi i32 [ %v143, %bb8 ], [ %v146, %bb9 ]
  %v151 = insertvalue { i32, i32 } undef, i32 %v149, 0
  %v152 = insertvalue { i32, i32 } %v151, i32 %v150, 1
  %v153 = extractvalue { i32, i32 } %v152, 0
  %v154 = zext i32 %v153 to i64
  %v155 = icmp eq i64 %v154, 0
  br i1 %v155, label %bb5, label %bb11
bb11:
  %v156 = icmp eq i64 %v154, 1
  br i1 %v156, label %bb4, label %bb3
bb12:
  br label %bb14
bb13:
  %v157 = trunc i64 %v61 to i32
  br label %bb14
bb14:
  %v158 = phi i32 [ 4294967295, %bb12 ], [ %v157, %bb13 ]
  %v159 = icmp ugt i32 %v47, 0
  %v160 = xor i1 %v159, 1
  br i1 %v160, label %bb9, label %bb8
bb15:
  %v277 = bitcast i8* %v19 to { float, float, float, float, float, float, float, float }*
  %v278 = getelementptr inbounds { float, float, float, float, float, float, float, float }, { float, float, float, float, float, float, float, float }* %v277, i32 0, i32 3
  %v161 = bitcast float* %v278 to i8*
  %v279 = bitcast i8* %v161 to float*
  %v162 = load float, float* %v279, align 4
  %v163 = fadd contract float %v127, %v162
  %v164 = fdiv contract float %v126, %v163
  %v165 = extractvalue { i8*, i64 } %v20, 0
  %v280 = bitcast i8* %v165 to float*
  %v281 = getelementptr inbounds float, float* %v280, i64 %v89
  %v166 = bitcast float* %v281 to i8*
  %v282 = bitcast i8* %v166 to float*
  %v167 = load float, float* %v282, align 4
  %v168 = fsub contract float %v167, %v164
  %v283 = bitcast i8* %v166 to float*
  store float %v168, float* %v283, align 4
  br label %bb2
}

define void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_max_i32_cuda_entry_a23ce822c89c7cd720reduce_workspace_maxINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB8_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCslfDnHtpJyg4_13vortx_shaders(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v0, i32 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i32, i32 } undef, i32 %v0, 0
  %v4 = insertvalue { i32, i32 } %v3, i32 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i32, i32 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb2, label %bb1
bb1:
  %v8 = extractvalue { i32, i32 } %v5, 0
  %v9 = extractvalue { i32, i32 } %v5, 1
  %v10 = call { i32, i32 } @_std__ops__Range_u32__as_std__iter__adapters__step_by__SpecRangeSetup_std__ops__Range_u32_____setup(i32 %v8, i32 %v9, i64 %v6) #0
  br label %bb3
bb2:
  call void asm sideeffect "trap;", ""()
  unreachable
bb3:
  %v12 = sub i64 %v6, 1
  %v13 = insertvalue { { i32, i32 }, i64, i1, [7 x i8] } undef, { i32, i32 } %v10, 0
  %v14 = insertvalue { { i32, i32 }, i64, i1, [7 x i8] } %v13, i64 %v12, 1
  %v15 = insertvalue { { i32, i32 }, i64, i1, [7 x i8] } %v14, i1 1, 2
  ret { { i32, i32 }, i64, i1, [7 x i8] } %v15
}

define i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCslfDnHtpJyg4_13vortx_shaders(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v13 = alloca i32, align 4
  %v4 = bitcast i32* %v13 to i8*
  %v14 = alloca i32, align 4
  %v5 = bitcast i32* %v14 to i8*
  %v15 = bitcast i8* %v4 to i32*
  store i32 %v2, i32* %v15, align 4
  %v16 = bitcast i8* %v5 to i32*
  store i32 %v3, i32* %v16, align 4
  %v6 = getelementptr i8, i8* %v5, i64 0
  %v7 = getelementptr i8, i8* %v4, i64 0
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(i8* %v6, i8* %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v17 = bitcast i8* %v4 to i32*
  %v10 = load i32, i32* %v17, align 4
  br label %bb4
bb3:
  %v18 = bitcast i8* %v5 to i32*
  %v11 = load i32, i32* %v18, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
}

define void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_summINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB6_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_u32_as_std__ops__Add___add(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_mulmINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB6_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_u32_as_std__ops__Mul___mul(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_mullINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB6_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_i32_as_std__ops__Mul___mul(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_sumfINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB6_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call float @_f32_as_std__ops__Add___add(float %v9, float %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, float %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_max_f32_cuda_entry_239487c8da53001520reduce_workspace_maxINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB8_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb5, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call float @__nv_fmaxf(float %v9, float %v12) #0
  br label %bb6
bb4:
  br label %bb5
bb5:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb6:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, float %v13) #0
  br label %bb4
bb7:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)

define { i32, i32, i32 } @khal_std__arch__cuda__global_invocation_id() alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb2
bb2:
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #0
  br label %bb3
bb3:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb4
bb4:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb5
bb5:
  %v5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0
  br label %bb6
bb6:
  %v6 = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %v0, i32 %v3) #0
  %v7 = extractvalue { i32, i1 } %v6, 1
  %v8 = xor i1 %v7, 1
  br i1 %v8, label %bb7, label %bb16
bb7:
  %v9 = extractvalue { i32, i1 } %v6, 0
  %v10 = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %v1, i32 %v4) #0
  %v11 = extractvalue { i32, i1 } %v10, 1
  %v12 = xor i1 %v11, 1
  br i1 %v12, label %bb8, label %bb17
bb8:
  %v13 = extractvalue { i32, i1 } %v10, 0
  %v14 = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %v2, i32 %v5) #0
  %v15 = extractvalue { i32, i1 } %v14, 1
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb9, label %bb18
bb9:
  %v17 = extractvalue { i32, i1 } %v14, 0
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb10
bb10:
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb11
bb11:
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z() #0
  br label %bb12
bb12:
  %v21 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %v9, i32 %v18) #0
  %v22 = extractvalue { i32, i1 } %v21, 1
  %v23 = xor i1 %v22, 1
  br i1 %v23, label %bb13, label %bb19
bb13:
  %v24 = extractvalue { i32, i1 } %v21, 0
  %v25 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %v13, i32 %v19) #0
  %v26 = extractvalue { i32, i1 } %v25, 1
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb14, label %bb20
bb14:
  %v28 = extractvalue { i32, i1 } %v25, 0
  %v29 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %v17, i32 %v20) #0
  %v30 = extractvalue { i32, i1 } %v29, 1
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb15, label %bb21
bb15:
  %v32 = extractvalue { i32, i1 } %v29, 0
  %v33 = insertvalue { i32, i32, i32 } undef, i32 %v24, 0
  %v34 = insertvalue { i32, i32, i32 } %v33, i32 %v28, 1
  %v35 = insertvalue { i32, i32, i32 } %v34, i32 %v32, 2
  ret { i32, i32, i32 } %v35
bb16:
  call void @llvm.trap() #0
  unreachable
bb17:
  call void @llvm.trap() #0
  unreachable
bb18:
  call void @llvm.trap() #0
  unreachable
bb19:
  call void @llvm.trap() #0
  unreachable
bb20:
  call void @llvm.trap() #0
  unreachable
bb21:
  call void @llvm.trap() #0
  unreachable
}

define void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_min_u32_cuda_entry_0e2d7d8e8e92801620reduce_workspace_minINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB8_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCslfDnHtpJyg4_13vortx_shaders(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCslfDnHtpJyg4_13vortx_shaders(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v13 = alloca i32, align 4
  %v4 = bitcast i32* %v13 to i8*
  %v14 = alloca i32, align 4
  %v5 = bitcast i32* %v14 to i8*
  %v15 = bitcast i8* %v4 to i32*
  store i32 %v2, i32* %v15, align 4
  %v16 = bitcast i8* %v5 to i32*
  store i32 %v3, i32* %v16, align 4
  %v6 = getelementptr i8, i8* %v5, i64 0
  %v7 = getelementptr i8, i8* %v4, i64 0
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(i8* %v6, i8* %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v17 = bitcast i8* %v5 to i32*
  %v10 = load i32, i32* %v17, align 4
  br label %bb4
bb3:
  %v18 = bitcast i8* %v4 to i32*
  %v11 = load i32, i32* %v18, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
}

define void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_mulfINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB6_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call float @_f32_as_std__ops__Mul___mul(float %v9, float %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, float %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define void @_RINvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce20reduce_workspace_sumlINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB6_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_i32_as_std__ops__Add___add(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_min_i32_cuda_entry_ae0d0e5623f305d720reduce_workspace_minINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuflKj80_EEB8_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCslfDnHtpJyg4_13vortx_shaders(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define i32 @_RNvYlNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCslfDnHtpJyg4_13vortx_shaders(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v13 = alloca i32, align 4
  %v4 = bitcast i32* %v13 to i8*
  %v14 = alloca i32, align 4
  %v5 = bitcast i32* %v14 to i8*
  %v15 = bitcast i8* %v4 to i32*
  store i32 %v2, i32* %v15, align 4
  %v16 = bitcast i8* %v5 to i32*
  store i32 %v3, i32* %v16, align 4
  %v6 = getelementptr i8, i8* %v5, i64 0
  %v7 = getelementptr i8, i8* %v4, i64 0
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(i8* %v6, i8* %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v17 = bitcast i8* %v5 to i32*
  %v10 = load i32, i32* %v17, align 4
  br label %bb4
bb3:
  %v18 = bitcast i8* %v4 to i32*
  %v11 = load i32, i32* %v18, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
}

define void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_min_f32_cuda_entry_a1b9271d7e5a05b220reduce_workspace_minINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBuffKj80_EEB8_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb5, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call float @__nv_fminf(float %v9, float %v12) #0
  br label %bb6
bb4:
  br label %bb5
bb5:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb6:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, float %v13) #0
  br label %bb4
bb7:
  ret void
}

define void @_RINvNvNtNtCslfDnHtpJyg4_13vortx_shaders6linalg6reduce69cuda_oxide_kernel_246e25db_reduce_max_u32_cuda_entry_1f9cd5e87342aa3920reduce_workspace_maxINtNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glue7SmemBufmKj80_EEB8_(i64 %v0, i64 %v1, i8* %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i8* [ %v2, %entry ]
  %v6 = icmp ult i64 %v3, %v4
  %v7 = xor i1 %v6, 1
  br i1 %v7, label %bb6, label %bb1
bb1:
  %v8 = getelementptr i8, i8* %v5, i64 0
  %v9 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v8, i64 %v3) #0
  br label %bb2
bb2:
  %v10 = getelementptr i8, i8* %v5, i64 0
  %v11 = add i64 %v3, %v4
  %v12 = call i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v10, i64 %v11) #0
  br label %bb3
bb3:
  %v13 = call i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCslfDnHtpJyg4_13vortx_shaders(i32 %v9, i32 %v12) #0
  br label %bb4
bb4:
  call void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v5, i64 %v3, i32 %v13) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  call void asm sideeffect "bar.sync 0;", "~{memory}"() #0
  br label %bb7
bb7:
  ret void
}

define i32 @_RNvYmNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3maxCslfDnHtpJyg4_13vortx_shaders(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v13 = alloca i32, align 4
  %v4 = bitcast i32* %v13 to i8*
  %v14 = alloca i32, align 4
  %v5 = bitcast i32* %v14 to i8*
  %v15 = bitcast i8* %v4 to i32*
  store i32 %v2, i32* %v15, align 4
  %v16 = bitcast i8* %v5 to i32*
  store i32 %v3, i32* %v16, align 4
  %v6 = getelementptr i8, i8* %v5, i64 0
  %v7 = getelementptr i8, i8* %v4, i64 0
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(i8* %v6, i8* %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v17 = bitcast i8* %v4 to i32*
  %v10 = load i32, i32* %v17, align 4
  br label %bb4
bb3:
  %v18 = bitcast i8* %v5 to i32*
  %v11 = load i32, i32* %v18, align 4
  br label %bb4
bb4:
  %v12 = phi i32 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i32 %v12
}

define { i32, i32, i32, i32 } @vortx_shaders__linalg__shape__Shape__decompose(i8* %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i8* [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v80 = bitcast i8* %v2 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v81 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v80, i32 0, i32 1
  %v4 = bitcast i32* %v81 to i8*
  %v82 = bitcast i8* %v4 to i32*
  %v5 = load i32, i32* %v82, align 4
  %v83 = bitcast i8* %v2 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v84 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v83, i32 0, i32 2
  %v6 = bitcast i32* %v84 to i8*
  %v85 = bitcast i8* %v6 to i32*
  %v7 = load i32, i32* %v85, align 4
  %v8 = mul i32 %v5, %v7
  %v86 = bitcast i8* %v2 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v87 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v86, i32 0, i32 3
  %v9 = bitcast i32* %v87 to i8*
  %v88 = bitcast i8* %v9 to i32*
  %v10 = load i32, i32* %v88, align 4
  %v11 = mul i32 %v8, %v10
  %v12 = icmp eq i32 %v11, 0
  br i1 %v12, label %bb3, label %bb5
bb1:
  %v13 = phi i32 [ %v22, %bb4 ], [ %v27, %bb5 ]
  %v14 = phi i32 [ %v23, %bb4 ], [ %v28, %bb5 ]
  %v15 = insertvalue { i32, i32 } undef, i32 %v13, 0
  %v16 = insertvalue { i32, i32 } %v15, i32 %v14, 1
  %v17 = extractvalue { i32, i32 } %v16, 0
  %v18 = zext i32 %v17 to i64
  %v19 = icmp eq i64 %v18, 0
  br i1 %v19, label %bb8, label %bb2
bb2:
  %v20 = icmp eq i64 %v18, 1
  br i1 %v20, label %bb9, label %bb7
bb3:
  br label %bb4
bb4:
  %v21 = insertvalue { i32, i32 } undef, i32 0, 0
  %v22 = extractvalue { i32, i32 } %v21, 0
  %v23 = extractvalue { i32, i32 } %v21, 1
  br label %bb1
bb5:
  %v24 = udiv i32 %v3, %v11
  %v25 = insertvalue { i32, i32 } undef, i32 1, 0
  %v26 = insertvalue { i32, i32 } %v25, i32 %v24, 1
  %v27 = extractvalue { i32, i32 } %v26, 0
  %v28 = extractvalue { i32, i32 } %v26, 1
  br label %bb1
bb6:
  %v29 = phi i32 [ 0, %bb8 ], [ %v34, %bb9 ]
  %v30 = mul i32 %v29, %v11
  %v31 = sub i32 %v3, %v30
  %v32 = mul i32 %v7, %v10
  %v33 = icmp eq i32 %v32, 0
  br i1 %v33, label %bb12, label %bb14
bb7:
  unreachable
bb8:
  br label %bb6
bb9:
  %v34 = extractvalue { i32, i32 } %v16, 1
  br label %bb6
bb10:
  %v35 = phi i32 [ %v44, %bb13 ], [ %v49, %bb14 ]
  %v36 = phi i32 [ %v45, %bb13 ], [ %v50, %bb14 ]
  %v37 = insertvalue { i32, i32 } undef, i32 %v35, 0
  %v38 = insertvalue { i32, i32 } %v37, i32 %v36, 1
  %v39 = extractvalue { i32, i32 } %v38, 0
  %v40 = zext i32 %v39 to i64
  %v41 = icmp eq i64 %v40, 0
  br i1 %v41, label %bb16, label %bb11
bb11:
  %v42 = icmp eq i64 %v40, 1
  br i1 %v42, label %bb17, label %bb7
bb12:
  br label %bb13
bb13:
  %v43 = insertvalue { i32, i32 } undef, i32 0, 0
  %v44 = extractvalue { i32, i32 } %v43, 0
  %v45 = extractvalue { i32, i32 } %v43, 1
  br label %bb10
bb14:
  %v46 = udiv i32 %v31, %v32
  %v47 = insertvalue { i32, i32 } undef, i32 1, 0
  %v48 = insertvalue { i32, i32 } %v47, i32 %v46, 1
  %v49 = extractvalue { i32, i32 } %v48, 0
  %v50 = extractvalue { i32, i32 } %v48, 1
  br label %bb10
bb15:
  %v51 = phi i32 [ 0, %bb16 ], [ %v55, %bb17 ]
  %v52 = mul i32 %v51, %v32
  %v53 = sub i32 %v31, %v52
  %v54 = icmp eq i32 %v10, 0
  br i1 %v54, label %bb20, label %bb22
bb16:
  br label %bb15
bb17:
  %v55 = extractvalue { i32, i32 } %v38, 1
  br label %bb15
bb18:
  %v56 = phi i32 [ %v65, %bb21 ], [ %v70, %bb22 ]
  %v57 = phi i32 [ %v66, %bb21 ], [ %v71, %bb22 ]
  %v58 = insertvalue { i32, i32 } undef, i32 %v56, 0
  %v59 = insertvalue { i32, i32 } %v58, i32 %v57, 1
  %v60 = extractvalue { i32, i32 } %v59, 0
  %v61 = zext i32 %v60 to i64
  %v62 = icmp eq i64 %v61, 0
  br i1 %v62, label %bb24, label %bb19
bb19:
  %v63 = icmp eq i64 %v61, 1
  br i1 %v63, label %bb25, label %bb7
bb20:
  br label %bb21
bb21:
  %v64 = insertvalue { i32, i32 } undef, i32 0, 0
  %v65 = extractvalue { i32, i32 } %v64, 0
  %v66 = extractvalue { i32, i32 } %v64, 1
  br label %bb18
bb22:
  %v67 = udiv i32 %v53, %v10
  %v68 = insertvalue { i32, i32 } undef, i32 1, 0
  %v69 = insertvalue { i32, i32 } %v68, i32 %v67, 1
  %v70 = extractvalue { i32, i32 } %v69, 0
  %v71 = extractvalue { i32, i32 } %v69, 1
  br label %bb18
bb23:
  %v72 = phi i32 [ 0, %bb24 ], [ %v79, %bb25 ]
  %v73 = mul i32 %v72, %v10
  %v74 = sub i32 %v53, %v73
  %v75 = insertvalue { i32, i32, i32, i32 } undef, i32 %v29, 0
  %v76 = insertvalue { i32, i32, i32, i32 } %v75, i32 %v51, 1
  %v77 = insertvalue { i32, i32, i32, i32 } %v76, i32 %v72, 2
  %v78 = insertvalue { i32, i32, i32, i32 } %v77, i32 %v74, 3
  ret { i32, i32, i32, i32 } %v78
bb24:
  br label %bb23
bb25:
  %v79 = extractvalue { i32, i32 } %v59, 1
  br label %bb23
}

define { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCslfDnHtpJyg4_13vortx_shaders(i64 %v0, i64 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i64, i64 } undef, i64 %v0, 0
  %v4 = insertvalue { i64, i64 } %v3, i64 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i64, i64 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb2, label %bb1
bb1:
  %v8 = extractvalue { i64, i64 } %v5, 0
  %v9 = extractvalue { i64, i64 } %v5, 1
  %v10 = call { i64, i64 } @_std__ops__Range_usize__as_std__iter__adapters__step_by__SpecRangeSetup_std__ops__Range_usize_____setup(i64 %v8, i64 %v9, i64 %v6) #0
  br label %bb3
bb2:
  call void asm sideeffect "trap;", ""()
  unreachable
bb3:
  %v12 = sub i64 %v6, 1
  %v13 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } undef, { i64, i64 } %v10, 0
  %v14 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } %v13, i64 %v12, 1
  %v15 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } %v14, i1 1, 2
  ret { { i64, i64 }, i64, i1, [7 x i8] } %v15
}

define void @vortx_shaders__linalg__contiguous__contiguous_impl(i32 %v0, i32 %v1, i32 %v2, i8* %v3, i8* %v4, i64 %v5, i8* %v6, i64 %v7, i32 %v8) alwaysinline #0 {
entry:
  %v9 = insertvalue { i32, i32, i32 } undef, i32 %v0, 0
  %v10 = insertvalue { i32, i32, i32 } %v9, i32 %v1, 1
  %v11 = insertvalue { i32, i32, i32 } %v10, i32 %v2, 2
  %v12 = insertvalue { i8*, i64 } undef, i8* %v4, 0
  %v13 = insertvalue { i8*, i64 } %v12, i64 %v5, 1
  %v14 = insertvalue { i8*, i64 } undef, i8* %v6, 0
  %v15 = insertvalue { i8*, i64 } %v14, i64 %v7, 1
  br label %bb0
bb0:
  %v16 = phi { i32, i32, i32 } [ %v11, %entry ]
  %v17 = phi i8* [ %v3, %entry ]
  %v18 = phi { i8*, i64 } [ %v13, %entry ]
  %v19 = phi { i8*, i64 } [ %v15, %entry ]
  %v20 = phi i32 [ %v8, %entry ]
  %v139 = alloca { i32, i32, i32 }, align 4
  %v21 = bitcast { i32, i32, i32 }* %v139 to i8*
  %v140 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v22 = bitcast { { i32, i32 }, i64, i1, [7 x i8] }* %v140 to i8*
  %v141 = bitcast i8* %v21 to { i32, i32, i32 }*
  store { i32, i32, i32 } %v16, { i32, i32, i32 }* %v141, align 4
  %v142 = bitcast i8* %v21 to { i32, i32, i32 }*
  %v143 = getelementptr inbounds { i32, i32, i32 }, { i32, i32, i32 }* %v142, i32 0, i32 0
  %v23 = bitcast i32* %v143 to i8*
  %v144 = bitcast i8* %v23 to i32*
  %v24 = load i32, i32* %v144, align 4
  %v145 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v146 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v145, i32 0, i32 0
  %v25 = bitcast i32* %v146 to i8*
  %v147 = bitcast i8* %v25 to i32*
  %v26 = load i32, i32* %v147, align 4
  %v148 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v149 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v148, i32 0, i32 1
  %v27 = bitcast i32* %v149 to i8*
  %v150 = bitcast i8* %v27 to i32*
  %v28 = load i32, i32* %v150, align 4
  %v29 = mul i32 %v26, %v28
  %v151 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v152 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v151, i32 0, i32 2
  %v30 = bitcast i32* %v152 to i8*
  %v153 = bitcast i8* %v30 to i32*
  %v31 = load i32, i32* %v153, align 4
  %v32 = mul i32 %v29, %v31
  %v154 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v155 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v154, i32 0, i32 3
  %v33 = bitcast i32* %v155 to i8*
  %v156 = bitcast i8* %v33 to i32*
  %v34 = load i32, i32* %v156, align 4
  %v35 = mul i32 %v32, %v34
  %v36 = insertvalue { i32, i32 } undef, i32 %v24, 0
  %v37 = insertvalue { i32, i32 } %v36, i32 %v35, 1
  %v38 = extractvalue { i32, i32 } %v37, 0
  %v39 = extractvalue { i32, i32 } %v37, 1
  %v40 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangemEE3newCslfDnHtpJyg4_13vortx_shaders(i32 %v38, i32 %v39, i64 8388480) #0
  %v158 = bitcast i8* %v22 to { { i32, i32 }, i64, i1, [7 x i8] }*
  store { { i32, i32 }, i64, i1, [7 x i8] } %v40, { { i32, i32 }, i64, i1, [7 x i8] }* %v158, align 8
  br label %bb8
bb1:
  %v41 = phi i32 [ %v119, %bb7 ], [ %v102, %bb8 ]
  %v42 = phi i32 [ %v120, %bb7 ], [ %v105, %bb8 ]
  %v43 = add i64 %v107, 1
  %v159 = alloca i64, align 8
  %v44 = bitcast i64* %v159 to i8*
  %v160 = bitcast i8* %v44 to i64*
  store i64 %v43, i64* %v160, align 8
  %v161 = bitcast i8* %v44 to { i64 }*
  %v45 = load { i64 }, { i64 }* %v161, align 8
  %v46 = extractvalue { i64 } %v45, 0
  %v47 = sub i64 %v46, 0
  %v48 = icmp ule i64 %v47, 0
  %v49 = add i64 %v47, 0
  %v50 = select i1 %v48, i64 %v49, i64 1
  %v51 = icmp eq i64 %v50, 1
  %v162 = alloca { i64 }, align 8
  %v52 = bitcast { i64 }* %v162 to i8*
  %v163 = bitcast i8* %v52 to { i64 }*
  store { i64 } %v45, { i64 }* %v163, align 8
  %v53 = getelementptr inbounds i8, i8* %v52, i64 0
  %v164 = bitcast i8* %v53 to { { i64 } }*
  %v54 = load { { i64 } }, { { i64 } }* %v164, align 8
  %v165 = alloca { { i64 } }, align 8
  %v55 = bitcast { { i64 } }* %v165 to i8*
  %v166 = bitcast i8* %v55 to { { i64 } }*
  store { { i64 } } %v54, { { i64 } }* %v166, align 8
  %v167 = bitcast i8* %v55 to i64*
  %v56 = load i64, i64* %v167, align 8
  %v57 = icmp ugt i64 %v56, 4294967295
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb14, label %bb13
bb2:
  unreachable
bb3:
  %v59 = extractvalue { i32, i32 } %v124, 1
  %v60 = mul i32 %v31, %v34
  %v61 = mul i32 %v28, %v31
  %v62 = mul i32 %v61, %v34
  %v63 = icmp eq i32 %v62, 0
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb5, label %bb16
bb4:
  ret void
bb5:
  %v65 = udiv i32 %v59, %v62
  %v66 = urem i32 %v59, %v62
  %v67 = icmp eq i32 %v60, 0
  %v68 = xor i1 %v67, 1
  br i1 %v68, label %bb6, label %bb17
bb6:
  %v69 = udiv i32 %v66, %v60
  %v70 = urem i32 %v59, %v60
  %v71 = icmp eq i32 %v34, 0
  %v72 = xor i1 %v71, 1
  br i1 %v72, label %bb7, label %bb18
bb7:
  %v73 = udiv i32 %v70, %v34
  %v74 = urem i32 %v59, %v34
  %v75 = zext i32 %v59 to i64
  %v168 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v169 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v168, i32 0, i32 4
  %v76 = bitcast i32* %v169 to i8*
  %v170 = bitcast i8* %v76 to i32*
  %v77 = load i32, i32* %v170, align 4
  %v78 = mul i32 %v65, %v77
  %v171 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v172 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v171, i32 0, i32 5
  %v79 = bitcast i32* %v172 to i8*
  %v173 = bitcast i8* %v79 to i32*
  %v80 = load i32, i32* %v173, align 4
  %v81 = mul i32 %v69, %v80
  %v82 = add i32 %v78, %v81
  %v174 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v175 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v174, i32 0, i32 6
  %v83 = bitcast i32* %v175 to i8*
  %v176 = bitcast i8* %v83 to i32*
  %v84 = load i32, i32* %v176, align 4
  %v85 = mul i32 %v73, %v84
  %v86 = add i32 %v82, %v85
  %v177 = bitcast i8* %v17 to { i32, i32, i32, i32, i32, i32, i32, i32 }*
  %v178 = getelementptr inbounds { i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i32, i32, i32, i32, i32, i32 }* %v177, i32 0, i32 7
  %v87 = bitcast i32* %v178 to i8*
  %v179 = bitcast i8* %v87 to i32*
  %v88 = load i32, i32* %v179, align 4
  %v89 = mul i32 %v74, %v88
  %v90 = add i32 %v86, %v89
  %v91 = add i32 %v20, %v90
  %v92 = zext i32 %v91 to i64
  %v93 = extractvalue { i8*, i64 } %v19, 1
  %v94 = icmp ult i64 %v92, %v93
  %v95 = extractvalue { i8*, i64 } %v19, 0
  %v180 = bitcast i8* %v95 to i32*
  %v181 = getelementptr inbounds i32, i32* %v180, i64 %v92
  %v96 = bitcast i32* %v181 to i8*
  %v182 = bitcast i8* %v96 to i32*
  %v97 = load i32, i32* %v182, align 4
  %v98 = extractvalue { i8*, i64 } %v18, 0
  %v183 = bitcast i8* %v98 to i32*
  %v184 = getelementptr inbounds i32, i32* %v183, i64 %v75
  %v99 = bitcast i32* %v184 to i8*
  %v185 = bitcast i8* %v99 to i32*
  store i32 %v97, i32* %v185, align 4
  br label %bb1
bb8:
  %v186 = bitcast i8* %v22 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v187 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v186, i32 0, i32 0
  %v100 = bitcast { i32, i32 }* %v187 to i8*
  %v188 = bitcast i8* %v100 to { i32, i32 }*
  %v189 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v188, i32 0, i32 0
  %v101 = bitcast i32* %v189 to i8*
  %v190 = bitcast i8* %v101 to i32*
  %v102 = load i32, i32* %v190, align 4
  %v191 = bitcast i8* %v22 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v192 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v191, i32 0, i32 0
  %v103 = bitcast { i32, i32 }* %v192 to i8*
  %v193 = bitcast i8* %v103 to { i32, i32 }*
  %v194 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v193, i32 0, i32 1
  %v104 = bitcast i32* %v194 to i8*
  %v195 = bitcast i8* %v104 to i32*
  %v105 = load i32, i32* %v195, align 4
  %v196 = bitcast i8* %v22 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v197 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v196, i32 0, i32 1
  %v106 = bitcast i64* %v197 to i8*
  %v198 = bitcast i8* %v106 to i64*
  %v107 = load i64, i64* %v198, align 8
  %v199 = bitcast i8* %v22 to { { i32, i32 }, i64, i1, [7 x i8] }*
  %v200 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, { { i32, i32 }, i64, i1, [7 x i8] }* %v199, i32 0, i32 2
  %v108 = bitcast i1* %v200 to i8*
  %v201 = bitcast i8* %v108 to i1*
  %v109 = load i1, i1* %v201, align 1
  br label %bb1
bb9:
  %v110 = add i32 %v41, %v130
  %v111 = sub i32 %v42, 1
  %v112 = insertvalue { i32, i32 } undef, i32 1, 0
  %v113 = insertvalue { i32, i32 } %v112, i32 %v41, 1
  %v114 = extractvalue { i32, i32 } %v113, 0
  %v115 = extractvalue { i32, i32 } %v113, 1
  br label %bb11
bb10:
  %v116 = insertvalue { i32, i32 } undef, i32 0, 0
  %v117 = extractvalue { i32, i32 } %v116, 0
  %v118 = extractvalue { i32, i32 } %v116, 1
  br label %bb11
bb11:
  %v119 = phi i32 [ %v110, %bb9 ], [ %v41, %bb10 ]
  %v120 = phi i32 [ %v111, %bb9 ], [ %v42, %bb10 ]
  %v121 = phi i32 [ %v114, %bb9 ], [ %v117, %bb10 ]
  %v122 = phi i32 [ %v115, %bb9 ], [ %v118, %bb10 ]
  %v123 = insertvalue { i32, i32 } undef, i32 %v121, 0
  %v124 = insertvalue { i32, i32 } %v123, i32 %v122, 1
  %v125 = extractvalue { i32, i32 } %v124, 0
  %v126 = zext i32 %v125 to i64
  %v127 = icmp eq i64 %v126, 0
  br i1 %v127, label %bb4, label %bb12
bb12:
  %v128 = icmp eq i64 %v126, 1
  br i1 %v128, label %bb3, label %bb2
bb13:
  br label %bb15
bb14:
  %v129 = trunc i64 %v56 to i32
  br label %bb15
bb15:
  %v130 = phi i32 [ 4294967295, %bb13 ], [ %v129, %bb14 ]
  %v131 = icmp ugt i32 %v42, 0
  %v132 = xor i1 %v131, 1
  br i1 %v132, label %bb10, label %bb9
bb16:
  call void @llvm.trap() #0
  unreachable
bb17:
  call void @llvm.trap() #0
  unreachable
bb18:
  call void @llvm.trap() #0
  unreachable
}

define { i32, i32, i32, i32 } @_glamx__UVec4_as_std__ops__Rem___rem(i32 %v0, i32 %v1, i32 %v2, i32 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7) #0 {
entry:
  %v8 = insertvalue { i32, i32, i32, i32 } undef, i32 %v0, 0
  %v9 = insertvalue { i32, i32, i32, i32 } %v8, i32 %v1, 1
  %v10 = insertvalue { i32, i32, i32, i32 } %v9, i32 %v2, 2
  %v11 = insertvalue { i32, i32, i32, i32 } %v10, i32 %v3, 3
  %v12 = insertvalue { i32, i32, i32, i32 } undef, i32 %v4, 0
  %v13 = insertvalue { i32, i32, i32, i32 } %v12, i32 %v5, 1
  %v14 = insertvalue { i32, i32, i32, i32 } %v13, i32 %v6, 2
  %v15 = insertvalue { i32, i32, i32, i32 } %v14, i32 %v7, 3
  br label %bb0
bb0:
  %v16 = phi { i32, i32, i32, i32 } [ %v11, %entry ]
  %v17 = phi { i32, i32, i32, i32 } [ %v15, %entry ]
  %v58 = alloca { i32, i32, i32, i32 }, align 4
  %v18 = bitcast { i32, i32, i32, i32 }* %v58 to i8*
  %v59 = alloca { i32, i32, i32, i32 }, align 4
  %v19 = bitcast { i32, i32, i32, i32 }* %v59 to i8*
  %v60 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v16, { i32, i32, i32, i32 }* %v60, align 4
  %v61 = bitcast i8* %v19 to { i32, i32, i32, i32 }*
  store { i32, i32, i32, i32 } %v17, { i32, i32, i32, i32 }* %v61, align 4
  %v62 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v63 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v62, i32 0, i32 0
  %v20 = bitcast i32* %v63 to i8*
  %v64 = bitcast i8* %v20 to i32*
  %v21 = load i32, i32* %v64, align 4
  %v65 = bitcast i8* %v19 to { i32, i32, i32, i32 }*
  %v66 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v65, i32 0, i32 0
  %v22 = bitcast i32* %v66 to i8*
  %v67 = bitcast i8* %v22 to i32*
  %v23 = load i32, i32* %v67, align 4
  %v24 = icmp eq i32 %v23, 0
  %v25 = xor i1 %v24, 1
  br i1 %v25, label %bb1, label %bb5
bb1:
  %v26 = urem i32 %v21, %v23
  %v68 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v69 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v68, i32 0, i32 1
  %v27 = bitcast i32* %v69 to i8*
  %v70 = bitcast i8* %v27 to i32*
  %v28 = load i32, i32* %v70, align 4
  %v71 = bitcast i8* %v19 to { i32, i32, i32, i32 }*
  %v72 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v71, i32 0, i32 1
  %v29 = bitcast i32* %v72 to i8*
  %v73 = bitcast i8* %v29 to i32*
  %v30 = load i32, i32* %v73, align 4
  %v31 = icmp eq i32 %v30, 0
  %v32 = xor i1 %v31, 1
  br i1 %v32, label %bb2, label %bb6
bb2:
  %v33 = urem i32 %v28, %v30
  %v74 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v75 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v74, i32 0, i32 2
  %v34 = bitcast i32* %v75 to i8*
  %v76 = bitcast i8* %v34 to i32*
  %v35 = load i32, i32* %v76, align 4
  %v77 = bitcast i8* %v19 to { i32, i32, i32, i32 }*
  %v78 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v77, i32 0, i32 2
  %v36 = bitcast i32* %v78 to i8*
  %v79 = bitcast i8* %v36 to i32*
  %v37 = load i32, i32* %v79, align 4
  %v38 = icmp eq i32 %v37, 0
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb3, label %bb7
bb3:
  %v40 = urem i32 %v35, %v37
  %v80 = bitcast i8* %v18 to { i32, i32, i32, i32 }*
  %v81 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v80, i32 0, i32 3
  %v41 = bitcast i32* %v81 to i8*
  %v82 = bitcast i8* %v41 to i32*
  %v42 = load i32, i32* %v82, align 4
  %v83 = bitcast i8* %v19 to { i32, i32, i32, i32 }*
  %v84 = getelementptr inbounds { i32, i32, i32, i32 }, { i32, i32, i32, i32 }* %v83, i32 0, i32 3
  %v43 = bitcast i32* %v84 to i8*
  %v85 = bitcast i8* %v43 to i32*
  %v44 = load i32, i32* %v85, align 4
  %v45 = icmp eq i32 %v44, 0
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb4, label %bb8
bb4:
  %v47 = urem i32 %v42, %v44
  %v48 = insertvalue { i32, i32, i32, i32 } undef, i32 %v26, 0
  %v49 = insertvalue { i32, i32, i32, i32 } %v48, i32 %v33, 1
  %v50 = insertvalue { i32, i32, i32, i32 } %v49, i32 %v40, 2
  %v51 = insertvalue { i32, i32, i32, i32 } %v50, i32 %v47, 3
  ret { i32, i32, i32, i32 } %v51
bb5:
  call void @llvm.trap() #0
  unreachable
bb6:
  call void @llvm.trap() #0
  unreachable
bb7:
  call void @llvm.trap() #0
  unreachable
bb8:
  call void @llvm.trap() #0
  unreachable
}

define i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v0, i64 %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i8* [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v9 = bitcast i8* %v2 to { i8 addrspace(3)* }*
  %v10 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v9, i32 0, i32 0
  %v4 = bitcast i8 addrspace(3)** %v10 to i8*
  %v11 = bitcast i8* %v4 to i8 addrspace(3)**
  %v5 = load i8 addrspace(3)*, i8 addrspace(3)** %v11, align 8
  %v6 = getelementptr i8, i8 addrspace(3)* %v5, i64 0
  %v12 = bitcast i8 addrspace(3)* %v6 to i32 addrspace(3)*
  %v13 = getelementptr inbounds i32, i32 addrspace(3)* %v12, i64 %v3
  %v7 = bitcast i32 addrspace(3)* %v13 to i8 addrspace(3)*
  br label %bb1
bb1:
  %v14 = bitcast i8 addrspace(3)* %v7 to i32 addrspace(3)*
  %v8 = load i32, i32 addrspace(3)* %v14, align 4
  ret i32 %v8
}

define void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuflKj80_EINtNtB4_5index19MaybeIndexUncheckedlE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v0, i64 %v1, i32 %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i8* [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i32 [ %v2, %entry ]
  %v9 = bitcast i8* %v3 to { i8 addrspace(3)* }*
  %v10 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v9, i32 0, i32 0
  %v6 = bitcast i8 addrspace(3)** %v10 to i8*
  %v11 = bitcast i8* %v6 to i8 addrspace(3)**
  %v7 = load i8 addrspace(3)*, i8 addrspace(3)** %v11, align 8
  %v12 = bitcast i8 addrspace(3)* %v7 to i32 addrspace(3)*
  %v13 = getelementptr inbounds i32, i32 addrspace(3)* %v12, i64 %v4
  %v8 = bitcast i32 addrspace(3)* %v13 to i8 addrspace(3)*
  br label %bb1
bb1:
  %v14 = bitcast i8 addrspace(3)* %v8 to i32 addrspace(3)*
  store i32 %v5, i32 addrspace(3)* %v14, align 4
  ret void
}

define { i32, i32 } @_std__ops__Range_u32__as_std__iter__adapters__step_by__SpecRangeSetup_std__ops__Range_u32_____setup(i32 %v0, i32 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i32, i32 } undef, i32 %v0, 0
  %v4 = insertvalue { i32, i32 } %v3, i32 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i32, i32 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  %v37 = alloca { i32, i32 }, align 4
  %v7 = bitcast { i32, i32 }* %v37 to i8*
  %v38 = bitcast i8* %v7 to { i32, i32 }*
  store { i32, i32 } %v5, { i32, i32 }* %v38, align 4
  %v39 = bitcast i8* %v7 to { i32, i32 }*
  %v40 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v39, i32 0, i32 0
  %v8 = bitcast i32* %v40 to i8*
  %v41 = bitcast i8* %v8 to i32*
  %v9 = load i32, i32* %v41, align 4
  %v42 = bitcast i8* %v7 to { i32, i32 }*
  %v43 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v42, i32 0, i32 1
  %v10 = bitcast i32* %v43 to i8*
  %v44 = bitcast i8* %v10 to i32*
  %v11 = load i32, i32* %v44, align 4
  %v12 = icmp ult i32 %v9, %v11
  %v13 = xor i1 %v12, 1
  br i1 %v13, label %bb2, label %bb1
bb1:
  %v45 = bitcast i8* %v7 to { i32, i32 }*
  %v46 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v45, i32 0, i32 0
  %v14 = bitcast i32* %v46 to i8*
  %v47 = bitcast i8* %v14 to i32*
  %v15 = load i32, i32* %v47, align 4
  %v48 = bitcast i8* %v7 to { i32, i32 }*
  %v49 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v48, i32 0, i32 1
  %v16 = bitcast i32* %v49 to i8*
  %v50 = bitcast i8* %v16 to i32*
  %v17 = load i32, i32* %v50, align 4
  %v18 = icmp ule i32 %v15, %v17
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb5, label %bb4
bb2:
  br label %bb3
bb3:
  %v20 = phi i64 [ 0, %bb2 ], [ %v25, %bb6 ]
  %v21 = icmp eq i64 %v6, 0
  %v22 = xor i1 %v21, 1
  br i1 %v22, label %bb7, label %bb11
bb4:
  %v23 = sub i32 %v17, %v15
  %v24 = zext i32 %v23 to i64
  br label %bb6
bb5:
  br label %bb6
bb6:
  %v25 = phi i64 [ %v24, %bb4 ], [ 0, %bb5 ]
  br label %bb3
bb7:
  %v26 = udiv i64 %v20, %v6
  %v27 = urem i64 %v20, %v6
  %v28 = icmp ugt i64 %v27, 0
  %v29 = xor i1 %v28, 1
  br i1 %v29, label %bb9, label %bb8
bb8:
  %v30 = add i64 %v26, 1
  br label %bb10
bb9:
  br label %bb10
bb10:
  %v31 = phi i64 [ %v30, %bb8 ], [ %v26, %bb9 ]
  %v32 = trunc i64 %v31 to i32
  %v51 = bitcast i8* %v7 to { i32, i32 }*
  %v52 = getelementptr inbounds { i32, i32 }, { i32, i32 }* %v51, i32 0, i32 1
  %v33 = bitcast i32* %v52 to i8*
  %v53 = bitcast i8* %v33 to i32*
  store i32 %v32, i32* %v53, align 4
  %v54 = bitcast i8* %v7 to { i32, i32 }*
  %v34 = load { i32, i32 }, { i32, i32 }* %v54, align 4
  ret { i32, i32 } %v34
bb11:
  call void @llvm.trap() #0
  unreachable
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(i8* %v0, i8* %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i8* [ %v0, %entry ]
  %v3 = phi i8* [ %v1, %entry ]
  %v7 = bitcast i8* %v2 to i32*
  %v4 = load i32, i32* %v7, align 4
  %v8 = bitcast i8* %v3 to i32*
  %v5 = load i32, i32* %v8, align 4
  %v6 = icmp slt i32 %v4, %v5
  ret i1 %v6
}

define i32 @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v0, i64 %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i8* [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v9 = bitcast i8* %v2 to { i8 addrspace(3)* }*
  %v10 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v9, i32 0, i32 0
  %v4 = bitcast i8 addrspace(3)** %v10 to i8*
  %v11 = bitcast i8* %v4 to i8 addrspace(3)**
  %v5 = load i8 addrspace(3)*, i8 addrspace(3)** %v11, align 8
  %v6 = getelementptr i8, i8 addrspace(3)* %v5, i64 0
  %v12 = bitcast i8 addrspace(3)* %v6 to i32 addrspace(3)*
  %v13 = getelementptr inbounds i32, i32 addrspace(3)* %v12, i64 %v3
  %v7 = bitcast i32 addrspace(3)* %v13 to i8 addrspace(3)*
  br label %bb1
bb1:
  %v14 = bitcast i8 addrspace(3)* %v7 to i32 addrspace(3)*
  %v8 = load i32, i32 addrspace(3)* %v14, align 4
  ret i32 %v8
}

define i32 @_u32_as_std__ops__Add___add(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %v2, i32 %v3) #0
  %v5 = extractvalue { i32, i1 } %v4, 1
  %v6 = xor i1 %v5, 1
  br i1 %v6, label %bb1, label %bb2
bb1:
  %v7 = extractvalue { i32, i1 } %v4, 0
  ret i32 %v7
bb2:
  call void @llvm.trap() #0
  unreachable
}

define void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBufmKj80_EINtNtB4_5index19MaybeIndexUncheckedmE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v0, i64 %v1, i32 %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i8* [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i32 [ %v2, %entry ]
  %v9 = bitcast i8* %v3 to { i8 addrspace(3)* }*
  %v10 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v9, i32 0, i32 0
  %v6 = bitcast i8 addrspace(3)** %v10 to i8*
  %v11 = bitcast i8* %v6 to i8 addrspace(3)**
  %v7 = load i8 addrspace(3)*, i8 addrspace(3)** %v11, align 8
  %v12 = bitcast i8 addrspace(3)* %v7 to i32 addrspace(3)*
  %v13 = getelementptr inbounds i32, i32 addrspace(3)* %v12, i64 %v4
  %v8 = bitcast i32 addrspace(3)* %v13 to i8 addrspace(3)*
  br label %bb1
bb1:
  %v14 = bitcast i8 addrspace(3)* %v8 to i32 addrspace(3)*
  store i32 %v5, i32 addrspace(3)* %v14, align 4
  ret void
}

define i32 @_u32_as_std__ops__Mul___mul(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %v2, i32 %v3) #0
  %v5 = extractvalue { i32, i1 } %v4, 1
  %v6 = xor i1 %v5, 1
  br i1 %v6, label %bb1, label %bb2
bb1:
  %v7 = extractvalue { i32, i1 } %v4, 0
  ret i32 %v7
bb2:
  call void @llvm.trap() #0
  unreachable
}

declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)

define i32 @_i32_as_std__ops__Mul___mul(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %v2, i32 %v3) #0
  %v5 = extractvalue { i32, i1 } %v4, 1
  %v6 = xor i1 %v5, 1
  br i1 %v6, label %bb1, label %bb2
bb1:
  %v7 = extractvalue { i32, i1 } %v4, 0
  ret i32 %v7
bb2:
  call void @llvm.trap() #0
  unreachable
}

define float @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE4readCslfDnHtpJyg4_13vortx_shaders(i8* %v0, i64 %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i8* [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v9 = bitcast i8* %v2 to { i8 addrspace(3)* }*
  %v10 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v9, i32 0, i32 0
  %v4 = bitcast i8 addrspace(3)** %v10 to i8*
  %v11 = bitcast i8* %v4 to i8 addrspace(3)**
  %v5 = load i8 addrspace(3)*, i8 addrspace(3)** %v11, align 8
  %v6 = getelementptr i8, i8 addrspace(3)* %v5, i64 0
  %v12 = bitcast i8 addrspace(3)* %v6 to float addrspace(3)*
  %v13 = getelementptr inbounds float, float addrspace(3)* %v12, i64 %v3
  %v7 = bitcast float addrspace(3)* %v13 to i8 addrspace(3)*
  br label %bb1
bb1:
  %v14 = bitcast i8 addrspace(3)* %v7 to float addrspace(3)*
  %v8 = load float, float addrspace(3)* %v14, align 4
  ret float %v8
}

define float @_f32_as_std__ops__Add___add(float %v0, float %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi float [ %v0, %entry ]
  %v3 = phi float [ %v1, %entry ]
  %v4 = fadd contract float %v2, %v3
  ret float %v4
}

define void @_RNvXNtCs5BeLyOWDfsJ_8khal_std15cuda_oxide_glueINtB2_7SmemBuffKj80_EINtNtB4_5index19MaybeIndexUncheckedfE5writeCslfDnHtpJyg4_13vortx_shaders(i8* %v0, i64 %v1, float %v2) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i8* [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi float [ %v2, %entry ]
  %v9 = bitcast i8* %v3 to { i8 addrspace(3)* }*
  %v10 = getelementptr inbounds { i8 addrspace(3)* }, { i8 addrspace(3)* }* %v9, i32 0, i32 0
  %v6 = bitcast i8 addrspace(3)** %v10 to i8*
  %v11 = bitcast i8* %v6 to i8 addrspace(3)**
  %v7 = load i8 addrspace(3)*, i8 addrspace(3)** %v11, align 8
  %v12 = bitcast i8 addrspace(3)* %v7 to float addrspace(3)*
  %v13 = getelementptr inbounds float, float addrspace(3)* %v12, i64 %v4
  %v8 = bitcast float addrspace(3)* %v13 to i8 addrspace(3)*
  br label %bb1
bb1:
  %v14 = bitcast i8 addrspace(3)* %v8 to float addrspace(3)*
  store float %v5, float addrspace(3)* %v14, align 4
  ret void
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_u32___lt(i8* %v0, i8* %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i8* [ %v0, %entry ]
  %v3 = phi i8* [ %v1, %entry ]
  %v7 = bitcast i8* %v2 to i32*
  %v4 = load i32, i32* %v7, align 4
  %v8 = bitcast i8* %v3 to i32*
  %v5 = load i32, i32* %v8, align 4
  %v6 = icmp ult i32 %v4, %v5
  ret i1 %v6
}

define float @_f32_as_std__ops__Mul___mul(float %v0, float %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi float [ %v0, %entry ]
  %v3 = phi float [ %v1, %entry ]
  %v4 = fmul contract float %v2, %v3
  ret float %v4
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

define i32 @_i32_as_std__ops__Add___add(i32 %v0, i32 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i32 [ %v1, %entry ]
  %v4 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %v2, i32 %v3) #0
  %v5 = extractvalue { i32, i1 } %v4, 1
  %v6 = xor i1 %v5, 1
  br i1 %v6, label %bb1, label %bb2
bb1:
  %v7 = extractvalue { i32, i1 } %v4, 0
  ret i32 %v7
bb2:
  call void @llvm.trap() #0
  unreachable
}

define { i64, i64 } @_std__ops__Range_usize__as_std__iter__adapters__step_by__SpecRangeSetup_std__ops__Range_usize_____setup(i64 %v0, i64 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i64, i64 } undef, i64 %v0, 0
  %v4 = insertvalue { i64, i64 } %v3, i64 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i64, i64 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  %v35 = alloca { i64, i64 }, align 8
  %v7 = bitcast { i64, i64 }* %v35 to i8*
  %v36 = bitcast i8* %v7 to { i64, i64 }*
  store { i64, i64 } %v5, { i64, i64 }* %v36, align 8
  %v37 = bitcast i8* %v7 to { i64, i64 }*
  %v38 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v37, i32 0, i32 0
  %v8 = bitcast i64* %v38 to i8*
  %v39 = bitcast i8* %v8 to i64*
  %v9 = load i64, i64* %v39, align 8
  %v40 = bitcast i8* %v7 to { i64, i64 }*
  %v41 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v40, i32 0, i32 1
  %v10 = bitcast i64* %v41 to i8*
  %v42 = bitcast i8* %v10 to i64*
  %v11 = load i64, i64* %v42, align 8
  %v12 = icmp ult i64 %v9, %v11
  %v13 = xor i1 %v12, 1
  br i1 %v13, label %bb2, label %bb1
bb1:
  %v43 = bitcast i8* %v7 to { i64, i64 }*
  %v44 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v43, i32 0, i32 0
  %v14 = bitcast i64* %v44 to i8*
  %v45 = bitcast i8* %v14 to i64*
  %v15 = load i64, i64* %v45, align 8
  %v46 = bitcast i8* %v7 to { i64, i64 }*
  %v47 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v46, i32 0, i32 1
  %v16 = bitcast i64* %v47 to i8*
  %v48 = bitcast i8* %v16 to i64*
  %v17 = load i64, i64* %v48, align 8
  %v18 = icmp ule i64 %v15, %v17
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb5, label %bb4
bb2:
  br label %bb3
bb3:
  %v20 = phi i64 [ 0, %bb2 ], [ %v24, %bb6 ]
  %v21 = icmp eq i64 %v6, 0
  %v22 = xor i1 %v21, 1
  br i1 %v22, label %bb7, label %bb11
bb4:
  %v23 = sub i64 %v17, %v15
  br label %bb6
bb5:
  br label %bb6
bb6:
  %v24 = phi i64 [ %v23, %bb4 ], [ 0, %bb5 ]
  br label %bb3
bb7:
  %v25 = udiv i64 %v20, %v6
  %v26 = urem i64 %v20, %v6
  %v27 = icmp ugt i64 %v26, 0
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb9, label %bb8
bb8:
  %v29 = add i64 %v25, 1
  br label %bb10
bb9:
  br label %bb10
bb10:
  %v30 = phi i64 [ %v29, %bb8 ], [ %v25, %bb9 ]
  %v49 = bitcast i8* %v7 to { i64, i64 }*
  %v50 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %v49, i32 0, i32 1
  %v31 = bitcast i64* %v50 to i8*
  %v51 = bitcast i8* %v31 to i64*
  store i64 %v30, i64* %v51, align 8
  %v52 = bitcast i8* %v7 to { i64, i64 }*
  %v32 = load { i64, i64 }, { i64, i64 }* %v52, align 8
  ret { i64, i64 } %v32
bb11:
  call void @llvm.trap() #0
  unreachable
}


@llvm.used = appending global [33 x i8*] [i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @contiguous_cuda_entry_6c8a092756f20c36 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64, i8*)* @contiguous_with_offset_cuda_entry_730898f1e627395f to i8*), i8* bitcast (void (i8*, i8*, i8*, i8*, i64, i8*, i64, i8*, i64)* @gemm_naive_cuda_entry_7c653b54ed65ef8f to i8*), i8* bitcast (void (i8*, i8*, i8*, i8*, i64, i8*, i64, i8*, i64)* @gemm_tiled_cuda_entry_21aa8599370d695c to i8*), i8* bitcast (void (i8*, i8*, i8*, i8*, i64, i8*, i64, i8*, i64)* @gemm_tiled_vec4_cuda_entry_f80a592a3ac0a3f3 to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64, i8*, i64, i8*, i64)* @gpu_adam_cuda_entry_1d28efac816be4b8 to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_add_cuda_entry_3172e2e34d229764 to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_copy_cuda_entry_9e8456c9ad6620bb to i8*), i8* bitcast (void (i8*, i8*, i8*, i8*, i64, i8*, i64)* @gpu_copy_with_offsets_cuda_entry_f31750be5f3ce13a to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_div_cuda_entry_b9c2b235c2039e1d to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_elu_backward_cuda_entry_251ddb1b5f024100 to i8*), i8* bitcast (void (i8*, i8*, i64)* @gpu_elu_cuda_entry_38a39043a6eb4b5c to i8*), i8* bitcast (void (i8*, i8*, i64)* @gpu_elu_vec4_cuda_entry_045fc3e50e2a11db to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_mul_cuda_entry_8999a608a654ff30 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64, i8*, i64, i8*, i64, i8*, i64, i8*, i64, i8*, i64)* @gpu_ppo_actor_grad_cuda_entry_5af55a16ff51d183 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64, i8*, i64, i8*, i64)* @gpu_ppo_value_grad_cuda_entry_71d0ca5bc19893ab to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_sub_cuda_entry_d42e3f2b5cf54553 to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_tanh_backward_cuda_entry_e1a55edc2786a0fd to i8*), i8* bitcast (void (i8*, i8*, i64)* @gpu_tanh_cuda_entry_28094766a61b17b8 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_add_f32_cuda_entry_bd874dedbb24ff35 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_add_i32_cuda_entry_23d17cc9dc93c126 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_add_u32_cuda_entry_22ce7ff7f5eb01d4 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_max_f32_cuda_entry_239487c8da530015 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_max_i32_cuda_entry_a23ce822c89c7cd7 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_max_u32_cuda_entry_1f9cd5e87342aa39 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_min_f32_cuda_entry_a1b9271d7e5a05b2 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_min_i32_cuda_entry_ae0d0e5623f305d7 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_min_u32_cuda_entry_0e2d7d8e8e928016 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_mul_f32_cuda_entry_e9d98ad8207eae6b to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_mul_i32_cuda_entry_aa0d5c1e4a461d98 to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_mul_u32_cuda_entry_453e89c2a77c7e7a to i8*), i8* bitcast (void (i8*, i8*, i64, i8*, i64)* @reduce_sq_norm_cuda_entry_b6de3497e124bb58 to i8*), i8* bitcast (void (i8*, i8*, i8*, i64, i8*, i64)* @repeat_cuda_entry_136129c3b46edf0e to i8*)], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_max_i32_cuda_entry_a23ce822c89c7cd7, !"kernel", i32 1}
!1 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_add_u32_cuda_entry_22ce7ff7f5eb01d4, !"kernel", i32 1}
!2 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_mul_u32_cuda_entry_453e89c2a77c7e7a, !"kernel", i32 1}
!3 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_mul_i32_cuda_entry_aa0d5c1e4a461d98, !"kernel", i32 1}
!4 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_add_f32_cuda_entry_bd874dedbb24ff35, !"kernel", i32 1}
!5 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_max_f32_cuda_entry_239487c8da530015, !"kernel", i32 1}
!6 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_min_u32_cuda_entry_0e2d7d8e8e928016, !"kernel", i32 1}
!7 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_mul_f32_cuda_entry_e9d98ad8207eae6b, !"kernel", i32 1}
!8 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_sq_norm_cuda_entry_b6de3497e124bb58, !"kernel", i32 1}
!9 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_add_i32_cuda_entry_23d17cc9dc93c126, !"kernel", i32 1}
!10 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_min_i32_cuda_entry_ae0d0e5623f305d7, !"kernel", i32 1}
!11 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_min_f32_cuda_entry_a1b9271d7e5a05b2, !"kernel", i32 1}
!12 = !{void (i8*, i8*, i64, i8*, i64)* @reduce_max_u32_cuda_entry_1f9cd5e87342aa39, !"kernel", i32 1}
!13 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_add_cuda_entry_3172e2e34d229764, !"kernel", i32 1}
!14 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_copy_cuda_entry_9e8456c9ad6620bb, !"kernel", i32 1}
!15 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_div_cuda_entry_b9c2b235c2039e1d, !"kernel", i32 1}
!16 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_mul_cuda_entry_8999a608a654ff30, !"kernel", i32 1}
!17 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_sub_cuda_entry_d42e3f2b5cf54553, !"kernel", i32 1}
!18 = !{void (i8*, i8*, i8*, i8*, i64, i8*, i64)* @gpu_copy_with_offsets_cuda_entry_f31750be5f3ce13a, !"kernel", i32 1}
!19 = !{void (i8*, i8*, i64)* @gpu_tanh_cuda_entry_28094766a61b17b8, !"kernel", i32 1}
!20 = !{void (i8*, i8*, i64)* @gpu_elu_cuda_entry_38a39043a6eb4b5c, !"kernel", i32 1}
!21 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_elu_backward_cuda_entry_251ddb1b5f024100, !"kernel", i32 1}
!22 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @gpu_tanh_backward_cuda_entry_e1a55edc2786a0fd, !"kernel", i32 1}
!23 = !{void (i8*, i8*, i64)* @gpu_elu_vec4_cuda_entry_045fc3e50e2a11db, !"kernel", i32 1}
!24 = !{void (i8*, i8*, i8*, i8*, i64, i8*, i64, i8*, i64)* @gemm_tiled_vec4_cuda_entry_f80a592a3ac0a3f3, !"kernel", i32 1}
!25 = !{void (i8*, i8*, i8*, i8*, i64, i8*, i64, i8*, i64)* @gemm_tiled_cuda_entry_21aa8599370d695c, !"kernel", i32 1}
!26 = !{void (i8*, i8*, i8*, i8*, i64, i8*, i64, i8*, i64)* @gemm_naive_cuda_entry_7c653b54ed65ef8f, !"kernel", i32 1}
!27 = !{void (i8*, i8*, i64, i8*, i64, i8*, i64, i8*, i64, i8*, i64, i8*, i64, i8*, i64)* @gpu_ppo_actor_grad_cuda_entry_5af55a16ff51d183, !"kernel", i32 1}
!28 = !{void (i8*, i8*, i64, i8*, i64, i8*, i64, i8*, i64)* @gpu_ppo_value_grad_cuda_entry_71d0ca5bc19893ab, !"kernel", i32 1}
!29 = !{void (i8*, i8*, i64, i8*, i64, i8*)* @contiguous_with_offset_cuda_entry_730898f1e627395f, !"kernel", i32 1}
!30 = !{void (i8*, i8*, i64, i8*, i64)* @contiguous_cuda_entry_6c8a092756f20c36, !"kernel", i32 1}
!31 = !{void (i8*, i8*, i8*, i64, i8*, i64)* @repeat_cuda_entry_136129c3b46edf0e, !"kernel", i32 1}
!32 = !{void (i8*, i8*, i8*, i64, i8*, i64, i8*, i64, i8*, i64)* @gpu_adam_cuda_entry_1d28efac816be4b8, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32}

!nvvmir.version = !{!33}
!33 = !{i32 2, i32 0, i32 3, i32 1}
