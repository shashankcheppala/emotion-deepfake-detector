// Scroll reveals using IntersectionObserver + a tiny GSAP enter if available
const $ = (q, el=document) => el.querySelector(q);
const $$ = (q, el=document) => [...el.querySelectorAll(q)];

// Year
$('#year') && ($('#year').textContent = new Date().getFullYear());

// Reveal on scroll
const io = new IntersectionObserver((entries)=>{
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('is-visible');
      io.unobserve(e.target);
    }
  })
},{threshold: 0.12});
$$('.reveal').forEach(el => io.observe(el));

// Optional GSAP polish if loaded
if (window.gsap && window.ScrollTrigger){
  gsap.registerPlugin(ScrollTrigger);
  gsap.utils.toArray('.section').forEach((sec,i)=>{
    gsap.fromTo(sec, {opacity: 0, y: 18}, {opacity: 1, y:0, duration: .7, ease: 'power2.out',
      scrollTrigger: {trigger: sec, start: 'top 80%', toggleActions: 'play none none reverse'}});
  });
}
