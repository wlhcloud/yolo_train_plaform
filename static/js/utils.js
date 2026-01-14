

// 通用：预览图片
function previewImage(imgUrl) {
    layer.open({
        type: 1,
        title: false,
        closeBtn: 1,
        shadeClose: true,
        area: ['80%', '80%'],
        content: `<div style="text-align:center; padding:20px;">
                        <img src="${imgUrl}" class="img-fluid" alt="图片预览">
                    </div>`
    });
}
// 通用：设置左侧导航激活
function setNavActiveByKey(activeKey) {
    const navLinks = document.querySelectorAll('#projectNav .nav-link');

    navLinks.forEach(link => {
        link.classList.remove('active');
    });

    const activeLink = document.querySelector(`#projectNav .nav-link[data-nav-key="${activeKey}"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }
}