<template>
<head>
    <title>清华大学校园网 Tsinghua University Network</title>
    <link rel="shortcut icon" href="../image/icon16-yellow.ico" type="x-icon" />
    <link rel="icon" href="../image/icon32.png" sizes="32x32" type="png" />
  </head>

  <body id="popup">
    <!-- 此处popup沿用清华大学校园网网站写法 -->
    <div id="center">
      <div class="greet">欢迎，&nbsp;Welcome,</div>
      <div class="triangle_greet"></div>
      <div class="corner"></div>
      <div id="info">
        <h2 class="user_name" id="uname">JC</h2>
        <div id="info_content">
          <div class="label_text">已连接<p class="english">Duration</p></div>
          <div class="content_text"><span id="clock" style="color:#ec6677">00:00:00</span><span class="unit"></span></div>
          <div class="label_text"><a href='../TsinghuaNet.html'>已用流量</a><p class="english">Usage</p></div>
          <div class="usage_bar">
            <ul class="calibration"><li>125</li><li>100</li><li>75</li><li>50</li></ul>
            <div id="usage_value" style="width:17px;"></div>
            <div class="stripe"></div>
            <div class="stripe second_stripe"></div>
            <div class="content_text"><span class="unit" id="usage_flux">{{msg}}G</span></div>
          </div>
        </div>
      </div>
      <div class="bar">
<a href="#/">     
	<button class="disconnect" @click=" this.postReponse();">断开连接&nbsp;Disconnect</button>
</a>
      </div>
      <ul id="links">
        <li><a id="link_info" href="http://info.tsinghua.edu.cn" target="_new" title="清华信息门户&#10;info.tsinghua">Info</a></li>
        <li><a id="link_lib" href="http://lib.tsinghua.edu.cn/" target="_new" title="清华图书馆&#10;lib.tsinghua">Lib</a></li>
        <li><a id="link_learn" href="http://learn.tsinghua.edu.cn/" target="_new" title="清华网络学堂&#10;learn.tsinghua">Learn</a></li>
        <li><a id="link_mail" href="http://mail.tsinghua.edu.cn/" target="_new" title="清华邮箱&#10;mail.tsinghua">Mail</a></li>
      </ul>

    </div>
  </body>

</template>

<style >
@charset "utf-8";

a {
	color:#AD3B23;
}

body { 
	background:#DADBDC url(../image/background.png) repeat-x top;

}
#head {
	background:url(../image/wireless/logo.gif) no-repeat;
}

#title { 
	background:url(../image/wireless/popup_greeting.png) no-repeat; margin-left: 82px 
}

.greet{
	position:absolute; 
	top:0; left:0; width:720px; height:304px; 
	/* background:#E64E2E url('popup_greeting.png') no-repeat 32px 28px;  */
	-webkit-box-shadow:0 0 8px rgba(0,0,0,0.1); 
	-moz-box-shadow:0 0 4px rgba(0,0,0,0.1);
	box-shadow:0 0 8px rgba(0,0,0,0.1);
}

.triangle_greet{
	position:absolute; 
	top:304px; left:32px; border-color:#E64E2E transparent; border-width: 48px 48px 0 0;
}

#popup{
	min-width:640px; 
	min-height:480px; 
	background:#E6E6E6 url(../image/wireless/popup_bg.png) no-repeat top right;
}

#popup .greet{
	top:64px; 
	left:48px; 
	width:512px; 
	height:224px; 
	background:#E64E2E url(../image/wireless/popup_greeting.png) no-repeat 30px 46px; 
}

#info .user_name{
	float:left; 
	color:#E64E2E; 
	font-family:Helvetica, Arial, sans-serif; 
	font-size:34px; 
	font-weight:lighter;
}

.usage_bar #usage_value{
	float:left; 
	height:32px; 
	max-width:280px; 
	background:#FBB03B;
}

#links #link_info{background:url(../image/wireless/popup_info.gif) no-repeat 2px 2px;}
#links #link_lib{background:url(../image/wireless/popup_lib.gif) no-repeat 2px 2px;}
#links #link_learn{background:url(../image/wireless/popup_learn.gif) no-repeat 4px 1px;}
#links #link_mail{background:url(../image/wireless/popup_mail.gif) no-repeat 1px;}

#phone #head{
	position:absolute; top:0; left:0; margin:0; background:#E64E2E; width:100%; height:108px; box-shadow:0 0 8px rgba(0,0,0,0.3); 
	-webkit-box-shadow:0 0 8px rgba(0,0,0,0.3); 
	-moz-box-shadow:0 0 4px rgba(0,0,0,0.3);
} 

#phone #head img{
	display:block;
	margin:8px 0 0 16px; 
	width:254px;
	}

#phone .triangle_head{
	position:absolute; 
	top:108px; left:4px;
	border-color:#E64E2E #E6E6E6; 
	border-width: 32px 32px 0 0;
}

/* ///JC */
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

#notice_box{position:absolute; top:128px; left:-16px; width:320px; height:172px;}
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

#connect:hover{background-color:#D2E042; background-position:center -47px;}
#connect:active{background-color:#AAC30B; background-position:center -94px;}
#connect:disable{background-color:#CCCFB7; background-position:center -141px;}

#account .english{font-size:11px;}

#notice .text{display:block; position:absolute; top:48px; left:48px; color:#FFF; width:240px; height:90px; overflow:auto;}

#system{position:absolute; bottom:0; left:-16px; width:320px; height:48px; background:#808080; -webkit-box-shadow:0 0 8px rgba(0,0,0,0.1); -moz-box-shadow:0 0 4px rgba(0,0,0,0.1); box-shadow:0 0 8px rgba(0,0,0,0.1);}
#system > li > a {position:absolute; bottom:16px; display:block; padding-top:36px; float:left; min-width:34px; height:16px; color:#FFF; font-size:13px; font-family:Helvetica,Arial, sans-serif;}
#system > li:last-child a{margin-right:0;}

#popup #center {position:relative; width:640px; height:480px; background:url(../image/popup_shadow.png) no-repeat 48px 344px;}
#popup .triangle_greet{left:80px; top:288px;}
#popup #info{top:96px; left:176px; padding:32px; width:352px; height:208px;}

#widget li{float:left; padding-left:36px; font-size:12px; color:#000;}
#widget li a{color:#000;}
#widget .detail a{color:#4F4F4F;}
#widget .detail{float:left; margin:2px 0 0 8px; line-height:1.3;}
#widget .english{font-size:11px; color:#4F4F4F;}
.contact .title{float:left;}

#widget .contact a{display:block;}

#foot{float:left; margin-top:64px; width:100%; height:100%; text-align:right; font-size:12px; color:#6E6F6F;}
#foot .english{font-size:11px;}

#info #account{float:left; margin:8px 0 0 16px; background-position:2px 4px; line-height:1.1;}
#info #info_content{float:left; margin-top:-8px;}
#info .label_text{clear:left; float:left; width:56px; height:32px; margin-top:22px; text-align:right; font-size:13px;}
#info .label_text .english{font-size:13px;}
#info .content_text{float:left; margin:24px 0 0 20px; padding-top:4px; height:28px; font-family:Helvetica, Arial, sans-serif; font-size:28px; font-weight:lighter;}
#info .content_text .unit{margin-left:8px;color:#5C5C5C;}
#info .usage_bar{position:relative; float:left; margin:24px 0 0 16px; padding:0; width:280px; height:32px; background:url(../image/usage_bg.gif) repeat-y;} 
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
#phone .label_text .english{color:inherit; font-size:15px; margin-left:4px;}
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
#phone #foot{position:absolute; margin:0; top:488px; right:0;}

.chart{
width: 600px;
height:400px;
display: block;
}
</style>

<script>
import axios from 'axios';
export default {
  name:'THUconnect',
  data(){
    return {msg : '1'}
  },
  methods : {
    getReponse(){
          const path = 'http://localhost:5000/THUconnect';
          axios.get(path)
          .then ((res) => { 
              console.log(res.data)
              this.msg = res.data;
          }) 
          .catch ((err) => {
              console.error(err);
          });
      },

      //POST FUnction
    postReponse(){
		var netdata = 1;
          const path = 'http://localhost:5000/THUconnect';
          axios.post(path,netdata)
          .then ((res) => { 
			// this.getReponse();
			this.msg = res.data;
          }) 
          .catch ((err) => {
              console.error(err);
          });   
    },},
  created(){
      this.getReponse();

  }
    
}
</script>
