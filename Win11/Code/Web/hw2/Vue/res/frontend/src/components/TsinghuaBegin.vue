<!-- <script>
// import HelloWorld from './components/HelloWorld.vue'
// import VueRouter from 'vue-router'
export default {
  name: 'App',
  components: {
   
  }
}


</script> -->
<template>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="viewport" content="width=device-width; initial-scale=1.0; maximum-scale=1.0;">
<title>清华大学无线校园网 Tsinghua University Wireless Network</title>
<link rel="shortcut icon" href="../image/icon16-yellow.ico" type="image/x-icon" />
<link rel="icon" href=" ../image/icon32.png" sizes="32x32" type="image/png" />
</head>
<body id="non_phone">
<div id="center">
<div id="head">
<div id="title">
<p>清华大学无线校园网</p>
<p class="english">Tsinghua University Wireless Network</p>
</div>
</div>
<!-- start -->
<div class="qrcode" style="margin-right:220px;">
<center>
<p><img src="../image/qrcode_for_258.jpg" width="100" height="100"></p>
<font size="2">欢迎关注“清华大学信息服务”</font>
</center>
</div>
<!-- end -->
<div id="content"><div id="attsinghua" style="    position: absolute;    top: -137px;    right: 0px;">
<a href="http://mobileapp.tsinghua.edu.cn" target="_blank">
<img src="../image/wireless/attsinghua.png" style="    width: 215px;    height: 105px;">
</a></div>      <div class="greet">您好，&nbsp;Hello,</div>
<div class="triangle_greet"></div>
<div id="notice_box">        <div class="corner"></div>        <div id="notice">
<p>&nbsp;<br/><br/>&nbsp;</p>          <div width="320" align="center">
<marquee onmouseover="this.stop()" onmouseout="this.start()" scrollAmount="1.0" width="300" align="left" height="90" direction="up" border="0" id="msg" style="color:white;">
</marquee>
</div>
</div>
</div>
<div id="login">
<form name="login_form" id="login_form" action="/do_login.php" method="post" onkeydown="if(event.keyCode==13)do_login();" onsubmit="do_login();">
<input type="hidden" name="is_pad" value="2">
<input type="hidden" name="n" value="100">
 <input type="hidden" name="type" value="1">
 <input type="hidden" name="drop" value="0">
 <input type="hidden" name="action" value="login">
 <input type="hidden" name="ac_id" id="ac_id" value="1">
 <div class="field">
 <div class="label_text" for="uname">用户名{{msg}}<p class="english">User&nbsp;ID</p></div>
 <input type="text" name="uname" id="uname" value="" autocorrect="off" autocapitalize="off">
 </div>          <div class="field">
 <div class="label_text" for="pass">密码<p class="english">Password</p></div>
 <input type="password" name="pass" id="pass" autocorrect="off" autocapitalize="off">
 </div>
 <!-- <a id="account" href="https://usereg.tsinghua.edu.cn" title="账户设置&#10;Account&nbsp;Settings">账户设置
 <p class="english">Account&nbsp;Settings</p></a>   -->       
 <a href="#/connect"><input type="button" name="connect" id="connect" onclick=" ;" >  </a>     </form>      </div>      
 <div class="triangle_login"></div>      <div id="download">        <ul id="system">          
  <!--li><a class="windows" href="/files/Tunet2015.exe" title="Windows">Windows</a></li>         
   <li><a class="mac" href="/files/TUNet_mac_os_x86.zip" title="MacOS">MacOS</a></li>          
   <li><a class="linux" href="/files/TUNet_linux.tar.gz" title="Linux">Linux</a></li>          
   <li><a class="android" href="/files/Tunet2015.apk" title="Android">Android</a></li>          
   <li><a class="ios" href="/files/TUNet_IOS.zip" title="iOS">iOS</a></li-->          
   <li><a class="windows"  title="Windows">Windows</a></li>          
   <!--li><a class="mac"  title="MacOS">MacOS</a></li-->          
   <li><a class="mac"  title="MacOS">MacOS</a></li>          
   <li><a class="linux"  title="Linux">Linux</a></li>          
   <li><a class="android"  title="Android">Android</a></li>          
   <li><a class="ios"  title="iOS">iOS</a></li>        </ul>      </div>      
   <ul id="widget">        
    <li class="bookmark"><a href="javascript:addBookmark('TUNet 无线网登录');" title="收藏&#10;Bookmark&nbsp;This&nbsp;Page">收藏<p class="english">Bookmark&nbsp;This&nbsp;Page</p></a></li>        
    <li class="help"><a href="/help/help.html" title="帮助&#10;Help">帮助<p class="english">Help</p></a></li>        
    <li class="contact">        <div class="title">联系<p class="english">Contact</p></div>        
    <div class="detail english">+86-10-62784859<a href="mailto:its@tsinghua.edu.cn" title="电子邮件&#10;Email">its@tsinghua.edu.cn</a></div></li>      
    </ul>    </div>    <div id="foot">清华大学信息化技术中心<p class="english">Information Technology Center of Tsinghua University</p></div>  
    </div>
    </body>

</template>



<style>
@charset "utf-8";

a{color:#AD3B23;}

body { background:#DADBDC url(../image/wireless/background.png) repeat-x top;}
#head {background:url(../image/wireless/logo.gif) no-repeat;}
#title { background:url(../image/wireless/title.png) no-repeat; margin-left: 82px }

.greet{position:absolute; top:0; left:0; width:720px; height:304px; background:#E64E2E url(../image/wireless/greeting.png) no-repeat 32px 28px; -webkit-box-shadow:0 0 8px rgba(0,0,0,0.1); -moz-box-shadow:0 0 4px rgba(0,0,0,0.1); box-shadow:0 0 8px rgba(0,0,0,0.1);}

.triangle_greet{position:absolute; top:304px; left:32px; border-color:#E64E2E transparent; border-width: 48px 48px 0 0;}

/* #account{float:right; font-size:13px; padding-left:40px; background:url(../image/wireless/account.gif) no-repeat 0 0; } */

#notice_box .corner{border-color:transparent #AD3B23; border-width: 16px 16px 0 0;}
#notice_box #notice{position:relative; width:320px; height:160px; border-width: 16px 16px 0 0; background:#E6562E url(../image/wireless/notice.png) no-repeat 14px 10px; -webkit-box-shadow:0 0 8px rgba(0,0,0,0.1); -moz-box-shadow:0 0 4px rgba(0,0,0,0.1); box-shadow:0 0 8px rgba(0,0,0,0.1);}

#popup{min-width:640px; min-height:480px; background:#E6E6E6 url(../image/wireless/popup_bg.png) no-repeat top right;}
#popup .greet{top:64px; left:48px; width:512px; height:224px; background:#E64E2E url(../image/wireless/popup_greeting.png) no-repeat 30px 46px; }

#info .user_name{float:left; color:#E64E2E; font-family:Helvetica, Arial, sans-serif; font-size:34px; font-weight:lighter;}
.usage_bar #usage_value{float:left; height:32px; max-width:280px; background:#FBB03B;}

#links #link_info{background:url(../image/wireless/popup_info.gif) no-repeat 2px 2px;}
#links #link_lib{background:url(../image/wireless/popup_lib.gif) no-repeat 2px 2px;}
#links #link_learn{background:url(../image/wireless/popup_learn.gif) no-repeat 4px 1px;}
#links #link_mail{background:url(../image/wireless/popup_mail.gif) no-repeat 0 1px;}

#phone #head{position:absolute; top:0; left:0; margin:0; background:#E64E2E; width:100%; height:108px; box-shadow:0 0 8px rgba(0,0,0,0.3); -webkit-box-shadow:0 0 8px rgba(0,0,0,0.3); -moz-box-shadow:0 0 4px rgba(0,0,0,0.3);}
#phone #head img{display:block; margin:8px 0 0 16px; width:254px;}
#phone .triangle_head{position:absolute; top:108px; left:4px; border-color:#E64E2E #E6E6E6; border-width: 32px 32px 0 0;}
/* #phone #account{float:left; margin:8px 0 8px 20px; padding-left:24px; background:url(../image/wireless/phone_account.gif) no-repeat 0 5px; font-size:14px;} */

/* 二维码 qrcode */
.qrcode { float:right; margin-top:15px; margin-right:50px; }
.qrcode img { width:129px; height:129px; border:none; }
.qrcode p { line-height:18px; color:#5d5d5d; text-align:center; font-size:3px; }

*{margin:0; padding:0; border:0;}
a{text-decoration:none;}
a:hover{text-decoration:underline;}
input[type=submit], button{text-indent:-999em; overflow:hidden; cursor:pointer;  -webkit-appearance:none; -moz-appearance:none; border-radius:0;  -webkit-border-radius:0; -moz-border-radius:0;}
input[type=submit][disabled=disabled], button:disabled{cursor:default;}
input[type=text], input[type=password]{outline:none; -webkit-appearance:none; -moz-appearance:none; border-radius:0;  -webkit-border-radius:0; -moz-border-radius:0;}
ul{list-style:none;}
.english{font-family:Verdana, Geneva, sans-serif;}
.corner, .triangle_greet, .triangle_login, .triangle_head{width:0; height:0; border-style: solid;}
.greet, #title, #popup .alert{text-indent:-999em; overflow:hidden;}

body{min-width:768px; font-family:"Microsoft YaHei", "SimHei", "Apple LiGothic Medium";}
#center{margin:auto; width:768px; height:100%;}
#head{float:left; margin:16px 0 32px -86px;}
#title{ display: block; width:320px; height:105px; }

#content{position:relative; float:left; clear:both; width:100%; height:496px; background:url(../image/content_shadow.png) no-repeat 0 416px;}

#notice_box{position:absolute; top:128px; left:-16px; width:320px; height:172px;}
#download{position:absolute; bottom:0; left:0; width:304px; height:112px; background:url(../image/wired/download.png) no-repeat 0 8px;}
#widget{position:absolute; bottom:0; right:0; width:464px; height:32px;}

#login #login_form{margin:32px;}
.field {float:left; margin-bottom:16px; width:100%;}
.field :first-child{position:relative;}
.label_text{float:left; height:48px; font-size:21px;}
.label_text .english{color:#303030; font-size:16px;}
#remember{margin-left:96px; width:auto; height:32px;}
#remember .checkbox_text{float:left; color:#4D4D4D; font-size:13px;}
.checkbox_text .english{font-size:11px;}

#login, #popup #info{position:absolute; top:32px; left:304px; width:464px; height:304px; background:#F2F2F2; -webkit-box-shadow:0 0 8px rgba(0,0,0,0.1); -moz-box-shadow:0 0 4px rgba(0,0,0,0.1); box-shadow:0 0 8px rgba(0,0,0,0.1);}
.triangle_login{position:absolute; top:336px; left:688px; border-color:#F2F2F2 transparent; border-width: 48px 48px 0 0;}

input[type=text], input[type=password]{float:right; width:298px; height:42px; border:#E6E6E6 solid 1px; padding:2px; background:#FFF; font-family:Helvetica, Arial, sans-serif; font-size:34px;}
input[type=text]:hover, input[type=password]:hover{border-color:#CCCCCC;}
input[type=text]:focus, input[type=password]:focus{border-color:#FBB03B; -webkit-box-shadow:0 0 8px rgba(251,176,59,0.8); -moz-box-shadow:0 0 4px rgba(251,176,59,0.8); box-shadow:0 0 8px rgba(251,176,59,0.8);}
input[type=text][disabled=disabled], input[type=password][disabled=disabled]{background:#F2F2F2;}
input[type=text].form-error, input[type=password].form-error{border-color:#ED1C24;}

input[type=checkbox]{float:left; margin:4px 4px 0 0;}

#connect{margin-top:16px; width:398px; height:46px; border:#9CB22E solid 1px; background:#C0CE55 url(../image/connect.png) no-repeat center 0; overflow:hidden;}
#connect:hover{background-color:#D2E042; background-position:center -47px;}
#connect:active{background-color:#AAC30B; background-position:center -94px;}
#connect:disable{background-color:#CCCFB7; background-position:center -141px;}

#account .english{font-size:11px;}

#notice .text{display:block; position:absolute; top:48px; left:48px; color:#FFF; width:240px; height:90px; overflow:auto;}

#system{position:absolute; bottom:0; left:-16px; width:320px; height:48px; background:#808080; -webkit-box-shadow:0 0 8px rgba(0,0,0,0.1); -moz-box-shadow:0 0 4px rgba(0,0,0,0.1); box-shadow:0 0 8px rgba(0,0,0,0.1);}
#system > li > a {position:absolute; bottom:16px; display:block; padding-top:36px; float:left; min-width:34px; height:16px; color:#FFF; font-size:13px; font-family:Helvetica,Arial, sans-serif;}
#system > li:last-child a{margin-right:0;}

.windows{left:16px; background:url(../image/wired/windows.gif) no-repeat 2px 0;}
.mac{left:80px; background:url(../image/wired/mac.gif) no-repeat 0 0;}
.linux{left:144px; background:url(../image/wired/linux.gif) no-repeat 0 0;}
.android{left:207px; background:url(../image/wired/android.gif) no-repeat 0 0;}
.ios{left:271px; background:url(../image/wired/ios.gif) no-repeat 0 0;}

#popup #center {position:relative; width:640px; height:480px; background:url(../image/popup_shadow.png) no-repeat 48px 344px;}
#popup .triangle_greet{left:80px; top:288px;}
#popup #info{top:96px; left:176px; padding:32px; width:352px; height:208px;}

#widget li{float:left; padding-left:36px; font-size:12px; color:#000;}
#widget li a{color:#000;}
#widget .detail a{color:#4F4F4F;}
#widget .detail{float:left; margin:2px 0 0 8px; line-height:1.3;}
#widget .english{font-size:11px; color:#4F4F4F;}
.contact .title{float:left;}

.bookmark{background:url(../image/wired/bookmark.gif) no-repeat 8px 0;}
#widget .help{background:url(../image/wired/help.gif) no-repeat 4px 0;}
#widget .contact{float:right; background:url(../image/wired/contact.gif) no-repeat 4px 0;}
#widget .contact a{display:block;}

#foot{float:left; margin-top:64px; width:100%; height:100%; text-align:right; font-size:12px; color:#6E6F6F;}
#foot .english{font-size:11px;}

#info #account{float:left; margin:8px 0 0 16px; background-position:2px 4px; line-height:1.1;}
#info #info_content{float:left; margin-top:-8px;}
#info .label_text{clear:left; float:left; width:56px; height:32px; margin-top:22px; text-align:right; font-size:13px;}
#info .label_text .english{font-size:13px;}
#info .content_text{float:left; margin:24px 0 0 20px; padding-top:4px; height:28px; font-family:Helvetica, Arial, sans-serif; font-size:28px; font-weight:lighter;}
#info .content_text .unit{margin-left:8px;color:#5C5C5C;}
#info .usage_bar{position:relative; float:left; margin:24px 0 0 16px; padding:0; width:280px; height:32px; background:url(../image/wired/usage_bg.gif) repeat-y;} 
.usage_bar .calibration{position:absolute; top:-16px; right:14px; font-family:Verdana, Geneva, sans-serif; font-size:13px; color:#969696;}
.calibration li{float:right; width:53px; text-align:right;}
.usage_bar .stripe{position:absolute; top:0; right:4px; width:4px; height:100%; background:#F2F2F2;}
.usage_bar .second_stripe{right:12px;}
#info .usage_bar .content_text{position:absolute; top:2px; left:4px; float:none; margin:0; padding:0;}
.usage_bar .content_text .unit{font-size:24px;}

#popup .corner{position:absolute; top:288px; left:592px; border-color:transparent #CCCCCC; border-width: 16px 0 0 16px;}
#popup .bar{position:absolute; top:304px; right:32px; width:208px; height:80px; background:#FFFFFF;  -webkit-box-shadow:0 0 8px rgba(0,0,0,0.1); -moz-box-shadow:0 0 4px rgba(0,0,0,0.1); box-shadow:0 0 8px rgba(0,0,0,0.1);}

.disconnect{margin:16px; width:174px; height:46px; border:#B2AABE solid 1px; background:#C0BDCC url(../image/disconnect.png) no-repeat center 0; overflow:hidden;}
.disconnect:hover{background-color:#D6D2E0; background-position:center -47px;}
.disconnect:active{background-color:#B1AAC4; background-position:center -94px;}
.disconnect:disable{background-color:#CFCFCF; background-position:center -141px;}

#popup #links{position:absolute; bottom:32px; right:32px; width:234px;}
#links li{float:left; margin-right:12px; font-size:11px; font-family:Verdana, Geneva, sans-serif;}
#links li:last-child{float:right; margin:0;}
#links li a{display:block; padding:16px 0 0 26px; height:16px;}

#phone{position:relative; min-width:320px; background:#E6E6E6 none;}
#phone #login_box{position:absolute; top:80px; left:0; width:100%;}
#phone #login{position:static; margin:0 0 0 16px; float:right; width:auto; height:224px; box-shadow:0 0 8px rgba(0,0,0,0.2); -webkit-box-shadow:0 0 8px rgba(0,0,0,0.2); -moz-box-shadow:0 0 4px rgba(0,0,0,0.2);}
#phone #login_form{margin:18px; width:auto;}
#phone .field {position:relative; margin:0 0 16px 0; width:100%;}
#phone .label_text{position:absolute; top:8px; left:24px; height:auto; font-size:16px; color:#9E9E9E;}
#phone .label_text .enunglish{color:inherit; font-size:15px; margin-left:4px;}
#phone #remember{margin:0 0 16px 0; height:16px;}
#phone #remember input[type=checkbox]{margin:0; width:20px; height:20px;}
#phone #login_form #remember .checkbox_text{margin:0 0 0 6px; font-size:14px; color:#6E6E6E;}
#phone #remember .checkbox_text .english{margin-left:4px; font-size:13px;}
#phone #login_form input[type=text], #phone #login_form input[type=password]{padding:0; width:100%; height:40px;}
#phone #connect{margin:0; width:100%; height:40px; background-image:none; text-indent:0; font-size:18px; color:#421A00;}
#phone .triangle_login{top:304px; left:inherit; right:16px; border-color:#F2F2F2 #E6E6E6; border-width: 32px 32px 0 0;}
#phone #account .english{font-size:13px;}

#phone #widget{bottom:inherit; top:408px; width:auto;}
#phone #widget li{height:16px; margin-right:16px; padding-left:16px; font-size:14px; color:#000;}
#phone #widget li a{color:#000;}
#phone #widget li .english{font-size:13px; color:#535353;}
#phone .detail{clear:both; float:right; margin:8px 8px 0 0; text-align:right; font-size:13px; line-height:1.4;}
#phone .detail a{display:block; color:#535353;}
/* #phone .bookmark{background:url(../image/wired/phone_bookmark.gif) no-repeat 0 0;} */
/* #phone .help{float:left; background:url(../image/wired/phone_help.gif) no-repeat 0 0;} */
/* #phone #widget .contact{background:url(../image/wired/phone_contact.gif) no-repeat 0 0; margin:0 8px 0 0;} */
#phone #foot{position:absolute; margin:0; top:488px; right:0;}

</style>

<script>
import axios from 'axios';
export default {
  name:'THUbegin',
  data(){
    return {msg : ''}
  },
  methods : {
    //Get Function
      getReponse(){
          const path = 'http://localhost:5000/THUbegin';
          axios.get(path)
          .then ((res) => { 
              console.log(res.data)
              this.msg = res.data;
          }) 
          .catch ((err) => {
              console.error(err);
          });
      },

  },
  created(){
      this.getReponse();
  }
}
</script>