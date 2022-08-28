const app = Vue.createApp({
    // template: '<h1>Hello World {{firstName}} {{lastName}}</h1>.',
    data(){
        return {
            firstName: 'Jacob',
            lastName: 'C',
            picture: 'https://i.stack.imgur.com/2oiTY.png',
            a: 'try',
        }
    },
    methods:{
        async getUser(){
            const res = await fetch('https://randomuser.me/api');
            const { results } = await res.json;
            console.log(results)
            
            console.log(this.firstName)
            this.firstName = 'ZXY';
            this.a = 'try1'
        },
    },
})

app.mount('#app')

