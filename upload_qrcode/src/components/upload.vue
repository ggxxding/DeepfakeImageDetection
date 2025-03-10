<template>
    <el-upload
        class="avatar-uploader"
        action="https://run.mocky.io/v3/9d059bf9-4660-45f2-925d-ce80ad6c4d15"
        :show-file-list="false"
        :on-success="handleAvatarSuccess"
        :before-upload="beforeAvatarUpload"
    >
        <img v-if="imageUrl" :src="imageUrl" class="avatar" />
        <el-icon v-else class="avatar-uploader-icon"><Plus /></el-icon>
    </el-upload>
</template>

<script lang="ts" setup>
    import { ref } from 'vue'
    import { ElMessage } from 'element-plus'
    import { Plus } from '@element-plus/icons-vue'
    import axios from 'axios'

    import type { UploadProps } from 'element-plus'

    const imageUrl = ref('https://fuss10.elemecdn.com/3/63/4e7f3a15429bfda99bce42a18cdd1jpeg.jpeg?imageMogr2/thumbnail/360x360/format/webp/quality/100')
    const fetchImage = () => {
        console.log('222')
        axios.get('http://127.0.0.1:5000/get_qrcode',{
            responseType: 'blob'
        }).then((response) => {
            const blob = new Blob([response.data], { type: 'image/png' });
            imageUrl.value = URL.createObjectURL(blob);
        }).catch((error) => {
            console.error(error)
        })
    }
    fetchImage();
    console.log('111');

    const handleAvatarSuccess: UploadProps['onSuccess'] = (
    response,
    uploadFile
    ) => {
    imageUrl.value = URL.createObjectURL(uploadFile.raw!)
    }

    const beforeAvatarUpload: UploadProps['beforeUpload'] = (rawFile) => {
    if (rawFile.type !== 'image/jpeg') {
        ElMessage.error('Avatar picture must be JPG format!')
        return false
    } else if (rawFile.size / 1024 / 1024 > 2) {
        ElMessage.error('Avatar picture size can not exceed 2MB!')
        return false
    }
    return true
    }
</script>

<style scoped>
    .avatar-uploader .avatar {
    width: 178px;
    height: 178px;
    display: block;
    }
    </style>

    <style>
    .avatar-uploader .el-upload {
    border: 1px dashed var(--el-border-color);
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: var(--el-transition-duration-fast);
    }

    .avatar-uploader .el-upload:hover {
    border-color: var(--el-color-primary);
    }

    .el-icon.avatar-uploader-icon {
    font-size: 28px;
    color: #8c939d;
    width: 178px;
    height: 178px;
    text-align: center;
    }
</style>
